# Overview

The files in this folder reproduces the blog post: [Exploratory Analysis of RLHF Transformers with TransformerLens](https://www.lesswrong.com/posts/Ky3WnDwQbLAucGrXf/exploratory-analysis-of-rlhf-transformers-with)

0. Navigate to this folder, making sure to use the right [uv](https://github.com/astral-sh/uv) dependencies
1. Install dependencies using uv: `uv sync`
2. Run `sbatch arc_gpu_job.slurm` (takes about 12min on a _Tesla V100-PCIE-16GB_ GPU)
