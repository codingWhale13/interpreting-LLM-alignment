import os
from functools import partial
from typing import List

import einops
import numpy as np
import plotly.express as px
import torch
import transformer_lens.utils as utils
import trlx
from datasets import load_dataset
from fancy_einsum import einsum
from torchtyping import TensorType as TT
from transformer_lens import ActivationCache, HookedTransformer
from transformers import AutoModelForCausalLM, pipeline
from trlx.data.configs import TRLConfig
from trlx.data.default_configs import default_ppo_config

IMAGE_COUNTER = 0
IMAGES = "images"
os.makedirs(IMAGES, exist_ok=True)


if torch.cuda.is_available():
    print("CUDA is available.")
    device = int(os.environ.get("LOCAL_RANK", 0))
else:
    print("CUDA is _not_ available.")
    device = "cpu"


# ===== HELPERS =====


def imsave(tensor, xaxis="", yaxis="", title="", **kwargs):
    global IMAGE_COUNTER
    IMAGE_COUNTER += 1

    fig = px.imshow(
        utils.to_numpy(tensor),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        labels={"x": xaxis, "y": yaxis},
        **kwargs,
    )

    file_format = "png"
    filename = f"{IMAGE_COUNTER}_{title.replace(' ', '_').lower()}.{file_format}"
    fig.write_image(os.path.join(IMAGES, filename), format=file_format)


def line(tensor, renderer=None, **kwargs):
    px.line(y=utils.to_numpy(tensor), **kwargs).show(renderer)


def two_lines(tensor1, tensor2, renderer=None, **kwargs):
    px.line(y=[utils.to_numpy(tensor1), utils.to_numpy(tensor2)], **kwargs).show(
        renderer
    )


def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(
        y=y, x=x, labels={"x": xaxis, "y": yaxis, "color": caxis}, **kwargs
    ).show(renderer)


# ========== Part I: Finetuning with TRLX ==========


def get_negative_score(scores):
    "Extract value associated with a negative sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["NEGATIVE"]


# NOTE from Nils: Author did not provide config file, so I'm using TRLX's default
# Instead of: default_config = yaml.safe_load(open("configs/ppo_config.yml"))
default_config = default_ppo_config()


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)

    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1

    sentiment_fn = pipeline(
        "sentiment-analysis",
        "lvwerra/distilbert-imdb",
        top_k=2,
        truncation=True,
        batch_size=256,
        device=device,
    )

    def reward_fn(samples: List[str], **kwargs) -> List[float]:
        sentiments = list(map(get_negative_score, sentiment_fn(samples)))
        return sentiments

    # Take few words off of movies reviews as prompts
    imdb = load_dataset("imdb", split="train+test")
    prompts = [" ".join(review.split()[:4]) for review in imdb["text"]]

    return trlx.train(
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=[
            "It's hard to believe the sequel to Avatar has actually come out. After 13 years and what feels like half-a-dozen delays"
        ]
        * 64,
        config=config,
    )


# NOTE from Nils: Only train if we have not saved a checkpoint in a previous run
path_base_model = os.path.abspath("base_model")
if os.path.exists(path_base_model):
    print(f"Using already trained base model from path {path_base_model}")
else:
    trainer = main()
    trainer.model.base_model.save_pretrained(path_base_model)


# ========== Part II: TransformerLens ==========

# Note that we save the base model (which is inside the model returned by TRLX).
# In order to load it into a HookedTransformer, we need this base model rather
# than the version that includes the additional value head (which TRLX itself
# constructs).

torch.set_grad_enabled(False)

source_model = AutoModelForCausalLM.from_pretrained("lvwerra/gpt2-imdb")
rlhf_model = AutoModelForCausalLM.from_pretrained(
    "curt-tigges/gpt2-negative-movie-reviews"
)

rlhf_model = AutoModelForCausalLM.from_pretrained(path_base_model)

hooked_source_model = HookedTransformer.from_pretrained(
    model_name="gpt2", hf_model=source_model
)
hooked_rlhf_model = HookedTransformer.from_pretrained(
    model_name="gpt2", hf_model=rlhf_model
)

# ========== Part II.1: Initial examination ==========
print("\n1) Initial examination")

example_prompt = "This movie was really"
example_answer = " good"

hooked_source_model.generate(example_prompt, max_new_tokens=10, temperature=0.0)
hooked_rlhf_model.generate(example_prompt, max_new_tokens=10, temperature=0.0)

utils.test_prompt(example_prompt, example_answer, hooked_source_model, prepend_bos=True)
utils.test_prompt(example_prompt, example_answer, hooked_rlhf_model, prepend_bos=True)

prompts = [
    # "This film was very",
    "This movie was really",
    # "This movie was quite"
]
answers = [
    # (" bad", " good"),
    (" bad", " good"),
    # (" bad", " good"),
]

# List of the token (ie an integer) corresponding to each answer, in the format (correct_token, incorrect_token)
answer_tokens = []
for i in range(len(prompts)):
    answer_tokens.append(
        (
            hooked_rlhf_model.to_single_token(answers[i][0]),
            hooked_rlhf_model.to_single_token(answers[i][1]),
        )
    )
answer_tokens = torch.tensor(answer_tokens).to(device)
print(prompts)
print(answers)

tokens = hooked_rlhf_model.to_tokens(prompts, prepend_bos=True)

# Run the models and cache all activations
source_logits, source_cache = hooked_source_model.run_with_cache(tokens)
rlhf_logits, rlhf_cache = hooked_rlhf_model.run_with_cache(tokens)


def logit_diff(logits, answer_tokens, per_prompt=False):
    # We only take the final logits
    final_logits = logits[:, -1, :]
    answer_logits = final_logits.gather(dim=-1, index=answer_tokens)
    answer_logit_diff = answer_logits[:, 0] - answer_logits[:, 1]
    if per_prompt:
        return answer_logit_diff
    else:
        return answer_logit_diff.mean()


print(
    "Logit difference in source model between 'bad' and 'good':",
    logit_diff(source_logits, answer_tokens, per_prompt=True),
)
original_average_logit_diff_source = logit_diff(source_logits, answer_tokens)
print(
    "Average logit difference in source model:",
    logit_diff(source_logits, answer_tokens).item(),
)

print(
    "Logit difference in RLHF model between 'bad' and 'good':",
    logit_diff(rlhf_logits, answer_tokens, per_prompt=True),
)
original_average_logit_diff_rlhf = logit_diff(rlhf_logits, answer_tokens)
print(
    "Average logit difference in RLHF model:",
    logit_diff(rlhf_logits, answer_tokens).item(),
)

# ========== Part II.2: Direct logit attribution ==========
print("\n2) Direct logit attribution")

# Here we get the unembedding vectors for the answer tokens
answer_residual_directions = hooked_rlhf_model.tokens_to_residual_directions(
    answer_tokens
)
print("Answer residual directions shape:", answer_residual_directions.shape)

logit_diff_directions = (
    answer_residual_directions[:, 0] - answer_residual_directions[:, 1]
)
print("Logit difference directions shape:", logit_diff_directions.shape)

# Cache syntax - resid_post is the residual stream at the end of the layer, -1 gets the final layer. The general syntax is [activation_name, layer_index, sub_layer_type].
final_residual_stream_source = source_cache["resid_post", -1]
final_residual_stream_rlhf = rlhf_cache["resid_post", -1]
print("Final residual stream shape:", final_residual_stream_rlhf.shape)
final_token_residual_stream_source = final_residual_stream_source[:, -1, :]
final_token_residual_stream_rlhf = final_residual_stream_rlhf[:, -1, :]

# Apply LayerNorm scaling
# pos_slice is the subset of the positions we take - here the final token of each prompt
scaled_final_token_residual_stream_source = source_cache.apply_ln_to_stack(
    final_token_residual_stream_source, layer=-1, pos_slice=-1
)
scaled_final_token_residual_stream_rlhf = rlhf_cache.apply_ln_to_stack(
    final_token_residual_stream_rlhf, layer=-1, pos_slice=-1
)

print("\nSource Model:")
average_logit_diff = einsum(
    "batch d_model, batch d_model -> ",
    scaled_final_token_residual_stream_source,
    logit_diff_directions,
) / len(prompts)
print("Calculated scaled average logit diff:", average_logit_diff.item())
print("Original logit difference:", original_average_logit_diff_source.item())

print("\nRLHF Model:")
average_logit_diff = einsum(
    "batch d_model, batch d_model -> ",
    scaled_final_token_residual_stream_rlhf,
    logit_diff_directions,
) / len(prompts)
print("Calculated scaled average logit diff:", average_logit_diff.item())
print("Original logit difference:", original_average_logit_diff_rlhf.item())

# ========== Part II.3: Logit Lens ==========
print("\n3) Logit Lens")

def residual_stack_to_logit_diff(
    residual_stack: TT["components", "batch", "d_model"], cache: ActivationCache
) -> float:
    scaled_residual_stack = rlhf_cache.apply_ln_to_stack(
        residual_stack, layer=-1, pos_slice=-1
    )
    return einsum(
        "... batch d_model, batch d_model -> ...",
        scaled_residual_stack,
        logit_diff_directions,
    ) / len(prompts)


accumulated_residual, labels = source_cache.accumulated_resid(
    layer=-1, incl_mid=True, pos_slice=-1, return_labels=True
)
logit_lens_logit_diffs = residual_stack_to_logit_diff(accumulated_residual, rlhf_cache)

accumulated_residual_rlhf, labels = rlhf_cache.accumulated_resid(
    layer=-1, incl_mid=True, pos_slice=-1, return_labels=True
)
logit_lens_logit_diffs_rlhf = residual_stack_to_logit_diff(
    accumulated_residual_rlhf, rlhf_cache
)

two_lines(
    logit_lens_logit_diffs,
    logit_lens_logit_diffs_rlhf,
    x=np.arange(hooked_rlhf_model.cfg.n_layers * 2 + 1) / 2,
    hover_name=labels,
    title="Logit Difference From Accumulated Residual Stream",
)

# ========== Part II.4: Layer attribution ==========
print("\n4) Layer attribution")

per_layer_residual, labels = source_cache.decompose_resid(
    layer=-1, pos_slice=-1, return_labels=True
)
per_layer_logit_diffs = residual_stack_to_logit_diff(per_layer_residual, source_cache)

per_layer_residual_rlhf, labels = rlhf_cache.decompose_resid(
    layer=-1, pos_slice=-1, return_labels=True
)
per_layer_logit_diffs_rlhf = residual_stack_to_logit_diff(
    per_layer_residual_rlhf, rlhf_cache
)

two_lines(
    per_layer_logit_diffs,
    per_layer_logit_diffs_rlhf,
    hover_name=labels,
    title="Logit Difference From Each Layer",
)

# ========== Part II.5: MLP activations ==========
print("\n5) MLP activations")

imsave(
    rlhf_cache["post", 10][0],
    yaxis="Pos",
    xaxis="Neuron",
    title="Neuron activations for single inputs",
    aspect="auto",
)

# ========== Part II.6: Model differences by attention head ==========
print("\n6) Model differences by attention head")

per_head_residual_source, labels = source_cache.stack_head_results(
    layer=-1, pos_slice=-1, return_labels=True
)
per_head_logit_diffs_source = residual_stack_to_logit_diff(
    per_head_residual_source, source_cache
)
per_head_logit_diffs_source = einops.rearrange(
    per_head_logit_diffs_source,
    "(layer head_index) -> layer head_index",
    layer=hooked_rlhf_model.cfg.n_layers,
    head_index=hooked_rlhf_model.cfg.n_heads,
)

per_head_residual_rlhf, labels = rlhf_cache.stack_head_results(
    layer=-1, pos_slice=-1, return_labels=True
)
per_head_logit_diffs_rlhf = residual_stack_to_logit_diff(
    per_head_residual_rlhf, rlhf_cache
)
per_head_logit_diffs_rlhf = einops.rearrange(
    per_head_logit_diffs_rlhf,
    "(layer head_index) -> layer head_index",
    layer=hooked_rlhf_model.cfg.n_layers,
    head_index=hooked_rlhf_model.cfg.n_heads,
)

per_head_model_diffs = per_head_logit_diffs_rlhf - per_head_logit_diffs_source

imsave(
    per_head_model_diffs,
    xaxis="Head",
    yaxis="Layer",
    title="Logit Difference From Each Head",
)

# ========== Part II.7: Activation patching for localization ==========
print("\n7) Activation patching for localization")

# We will use this function to patch different Parts of the residual stream
def patch_residual_component(
    to_residual_component: TT["batch", "pos", "d_model"],
    hook,
    subcomponent_index,
    from_cache,
):
    from_cache_component = from_cache[hook.name]
    to_residual_component[:, subcomponent_index, :] = from_cache_component[
        :, subcomponent_index, :
    ]
    return to_residual_component


# We will use this to patch specific heads
def patch_head_vector(
    rlhf_head_vector: TT["batch", "pos", "head_index", "d_head"],
    hook,
    subcomponent_index,
    from_cache,
):
    if isinstance(subcomponent_index, int):
        rlhf_head_vector[:, :, subcomponent_index, :] = from_cache[hook.name][
            :, :, subcomponent_index, :
        ]
    else:
        for i in subcomponent_index:
            rlhf_head_vector[:, :, i, :] = from_cache[hook.name][:, :, i, :]
    return rlhf_head_vector


def normalize_patched_logit_diff(patched_logit_diff):
    # Subtract corrupted logit diff to measure the improvement, divide by the total improvement from clean to corrupted to normalize
    # 0 means zero change, negative means more positive, 1 means equivalent to RLHF model, >1 means more negative than RLHF model
    return (patched_logit_diff - original_average_logit_diff_source) / (
        original_average_logit_diff_rlhf - original_average_logit_diff_source
    )


# Here we just take one of the example prompts and answers
tokens = hooked_rlhf_model.to_tokens(prompts, prepend_bos=True)

source_model_logits, source_model_cache = hooked_source_model.run_with_cache(
    tokens, return_type="logits"
)
rlhf_model_logits, rlhf_model_cache = hooked_rlhf_model.run_with_cache(
    tokens, return_type="logits"
)
source_model_average_logit_diff = logit_diff(source_model_logits, answer_tokens)
print("Source Model Average Logit Diff", source_model_average_logit_diff)
print("RLHF Model Average Logit Diff", original_average_logit_diff_rlhf)

# ========== Part II.8: Patch residual stream ==========
print("\n7) Patch residual stream")

patched_residual_stream_diff = torch.zeros(
    hooked_source_model.cfg.n_layers,
    tokens.shape[1],
    device=device,
    dtype=torch.float32,
)
for layer in range(hooked_source_model.cfg.n_layers):
    for position in range(tokens.shape[1]):
        hook_fn = partial(
            patch_residual_component,
            subcomponent_index=position,
            from_cache=rlhf_model_cache,
        )
        patched_logits = hooked_source_model.run_with_hooks(
            tokens,
            fwd_hooks=[(utils.get_act_name("resid_pre", layer), hook_fn)],
            return_type="logits",
        )
        patched_logit_diff = logit_diff(patched_logits, answer_tokens)

        patched_residual_stream_diff[layer, position] = normalize_patched_logit_diff(
            patched_logit_diff
        )


prompt_position_labels = [
    f"{tok}_{i}" for i, tok in enumerate(hooked_source_model.to_str_tokens(tokens[0]))
]
imsave(
    patched_residual_stream_diff,
    x=prompt_position_labels,
    title="Logit Difference From Patched Residual Stream",
    xaxis="Position",
    yaxis="Layer",
)

# ========== Part II.9: Patch MLPs & attention layers ==========
print("\n9) Path MLPs & activation layers")

patched_attn_diff = torch.zeros(
    hooked_source_model.cfg.n_layers,
    tokens.shape[1],
    device=device,
    dtype=torch.float32,
)
for layer in range(hooked_source_model.cfg.n_layers):
    for position in range(tokens.shape[1]):
        hook_fn = partial(
            patch_residual_component,
            subcomponent_index=position,
            from_cache=rlhf_model_cache,
        )

        # patch attention
        patched_logits = hooked_source_model.run_with_hooks(
            tokens,
            fwd_hooks=[(utils.get_act_name("attn_out", layer), hook_fn)],
            return_type="logits",
        )
        patched_attn_logit_diff = logit_diff(patched_logits, answer_tokens)
        # print(hooked_source_model.to_str_tokens(patched_logits.argmax(dim=2)[:,-1]))
        # print(f"Attention {layer=} {position=}")
        # print(hooked_source_model.to_str_tokens(patched_logits.argmax(dim=2)[:,-1]))

        patched_attn_diff[layer, position] = normalize_patched_logit_diff(
            patched_attn_logit_diff
        )


patched_mlp_diff = torch.zeros(
    hooked_source_model.cfg.n_layers,
    tokens.shape[1],
    device=device,
    dtype=torch.float32,
)
for layer in range(hooked_source_model.cfg.n_layers):
    for position in range(tokens.shape[1]):
        hook_fn = partial(
            patch_residual_component,
            subcomponent_index=position,
            from_cache=rlhf_model_cache,
        )

        # patch MLP
        patched_logits = hooked_source_model.run_with_hooks(
            tokens,
            fwd_hooks=[(utils.get_act_name("mlp_out", layer), hook_fn)],
            return_type="logits",
        )
        patched_mlp_logit_diff = logit_diff(patched_logits, answer_tokens)
        # print(f"MLP {layer=} {position=}")
        # print(hooked_source_model.to_str_tokens(patched_logits.argmax(dim=2)[:,-1]))

        patched_mlp_diff[layer, position] = normalize_patched_logit_diff(
            patched_mlp_logit_diff
        )

prompt_position_labels = [
    f"{tok}_{i}" for i, tok in enumerate(hooked_source_model.to_str_tokens(tokens[0]))
]
imsave(
    patched_attn_diff,
    x=prompt_position_labels,
    title="Logit Difference From Patched Attention Layers",
    xaxis="Position",
    yaxis="Layer",
)


prompt_position_labels = [
    f"{tok}_{i}" for i, tok in enumerate(hooked_source_model.to_str_tokens(tokens[0]))
]
imsave(
    patched_mlp_diff,
    x=prompt_position_labels,
    title="Logit Difference From Patched MLPs",
    xaxis="Position",
    yaxis="Layer",
)

# ========== Part II.10: Patch attention heads ==========
print("\n10) Patch attention heads")

patched_head_z_diff = torch.zeros(
    hooked_source_model.cfg.n_layers,
    hooked_source_model.cfg.n_heads,
    device=device,
    dtype=torch.float32,
)
for layer in range(hooked_source_model.cfg.n_layers):
    for head_index in range(hooked_source_model.cfg.n_heads):
        hook_fn = partial(
            patch_head_vector,
            subcomponent_index=head_index,
            from_cache=rlhf_model_cache,
        )
        patched_logits = hooked_source_model.run_with_hooks(
            tokens,
            fwd_hooks=[(utils.get_act_name("z", layer, "attn"), hook_fn)],
            return_type="logits",
        )
        patched_logit_diff = logit_diff(patched_logits, answer_tokens)
        # print(f"Attention {layer=} {head_index=}")
        # print(hooked_source_model.to_str_tokens(patched_logits.argmax(dim=2)[:,-1]))

        patched_head_z_diff[layer, head_index] = normalize_patched_logit_diff(
            patched_logit_diff
        )


imsave(
    patched_head_z_diff,
    title="Logit Difference From Patched Head Output",
    xaxis="Head",
    yaxis="Layer",
)

# ========== Part II.11: Patch multiple attention heads ==========
print("\n11) Patch multiple attention heads")

hook_fn = partial(
    patch_head_vector, subcomponent_index=(4, 9), from_cache=rlhf_model_cache
)
patched_logits = hooked_source_model.run_with_hooks(
    tokens,
    fwd_hooks=[(utils.get_act_name("z", 10, "attn"), hook_fn)],
    return_type="logits",
)
patched_logit_diff = normalize_patched_logit_diff(
    logit_diff(patched_logits, answer_tokens)
)

patched_logits.shape


print(logit_diff(patched_logits, answer_tokens))
print(patched_logit_diff)

print(hooked_source_model.to_str_tokens(patched_logits.argmax(dim=2)[-1]))
