# ParaFlow: Parallel Sampling for Flow Matching Models

<p>
  <a href="https://link.springer.com/chapter/10.1007/978-3-032-37038-9_8"><img src="https://img.shields.io/badge/ECCV-2026-blue" alt="ECCV 2026"></a>
  <a href="https://link.springer.com/chapter/10.1007/978-3-032-37038-9_8"><img src="https://img.shields.io/badge/Paper-Springer-green" alt="Paper"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-yellow" alt="License"></a>
</p>

Official PyTorch implementation of **ParaFlow**, accepted at **ECCV 2026**.

> **ParaFlow: Parallel Sampling for Flow Matching Models**
> Jianrong Lu, Bangwei Li, Haomin Zhang, Zhuoya Gu, Yongqing Lu, Jianhai Chen, Qinming He
> Computer Vision – ECCV 2026, LNCS, pp. 128–145
> [[Paper]](https://link.springer.com/chapter/10.1007/978-3-032-37038-9_8)

## Overview

ParaFlow is a **training-free** parallel sampling framework for flow matching text-to-image models.
It accelerates the inference of off-the-shelf models such as **FLUX.1** and **Stable Diffusion 3/3.5**
without any retraining, fine-tuning, or architectural changes.

The key ideas are:

* **Sliding-window parallel sampling.** Following [ParaDiGMS](https://arxiv.org/abs/2303.08435),
  ParaFlow maintains a sliding window over the denoising timesteps. Within one iteration, all steps
  in the window are solved together and the window advances past every step whose accumulated error
  stays below a given tolerance, turning the strictly sequential ODE solve into a small number of
  parallel iterations.
* **First-order velocity extrapolation.** Inside each window, only the first `num_anchor_steps`
  timesteps are evaluated with the transformer. The velocities of the remaining timesteps are
  approximated by a **first-order Taylor expansion of the velocity field** around the last anchor
  step, where the slope is estimated from the anchors of two consecutive windows.

This substantially reduces the number of expensive transformer evaluations while preserving image
quality, and it is complementary to the choice of ODE solver.

## Method

Each ParaFlow iteration performs the following steps:

1. **Anchor evaluation** – run the transformer on the first `num_anchor_steps` steps of the current
   window and keep the last one as the *anchor*.
2. **Extrapolation** – approximate the velocities of the remaining window steps by a first-order
   expansion around the anchor, using the slope estimated from the anchors of the previous and the
   current window.
3. **Parallel update** – integrate all window steps at once with the cumulative drift (Euler-style
   prefix sum).
4. **Error check & window shift** – measure the error against the previous iterate and advance the
   window past all steps that fall below `tolerance`.

The procedure stops when all `num_inference_steps` timesteps are below the tolerance or
`max_iterations` is reached.

## Installation

```bash
git clone https://github.com/Jianrong-Lu/ParaFlow.git
cd ParaFlow
pip install -r requirements.txt
```

> **Note:** FLUX.1-dev and Stable Diffusion 3/3.5 are gated models on Hugging Face.
> Please request access on the corresponding model pages and authenticate with
> `huggingface-cli login` before the first run.

## Quick Start

```python
import torch
from paraflow import ParaFlowFluxPipeline

pipe = ParaFlowFluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
)
pipe.to("cuda")

output, num_iterations = pipe(
    "A cat holding a sign that says hello world",
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,   # same schedule length as sequential sampling
    window_size=50,           # sliding-window size
    num_anchor_steps=1,       # transformer evaluations per window
    tolerance=0.01,           # error tolerance for advancing the window
    max_iterations=15,        # upper bound on parallel iterations
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0),
)
output.images[0].save("paraflow_flux.png")
print(f"Done in {num_iterations} parallel iterations.")
```

`ParaFlowSD3Pipeline` (Stable Diffusion 3 / 3.5) exposes the same interface.

## Reproducing the Paper Results

The `data/prompts.txt` file contains the 10k prompts used in our experiments.

### ParaFlow (parallel sampling)

```bash
# FLUX.1-dev, paper setting: 50 steps, window 50, 1 anchor step, tolerance 0.01
python generate_paraflow.py --model flux --num_images 5000

# Stable Diffusion 3 Medium / 3.5 Large
python generate_paraflow.py --model sd3   --num_images 5000
python generate_paraflow.py --model sd3.5 --num_images 5000
```

Each run creates a timestamped directory under `outputs/` containing the generated images and a
`result_log.txt` with the mean number of parallel iterations and the mean per-image latency.

### Sequential baselines

Standard sequential sampling with the Euler, UniPC, or DPM-Solver flow-matching schedulers:

```bash
python generate_sequential.py --model flux --sampler euler --num_inference_steps 50
python generate_sequential.py --model sd3  --sampler unipc --num_inference_steps 50
```

For the ParaDiGMS baseline, please refer to the official repository:
https://github.com/prowlive/paradigms

## Parameters

| Parameter             | Default (script) | Description                                                                                            |
| --------------------- | ---------------- | ------------------------------------------------------------------------------------------------------ |
| `num_inference_steps` | 50               | Number of denoising timesteps, identical to the sequential schedule.                                   |
| `window_size`         | 50               | Size of the sliding window processed in each iteration.                                                |
| `num_anchor_steps`    | 1                | Leading window steps evaluated with the transformer; the remaining steps use first-order extrapolation. |
| `tolerance`           | 0.01             | Relative error tolerance; controls how far the window advances per iteration.                          |
| `max_iterations`      | 15               | Upper bound on the number of parallel iterations.                                                      |

Larger `tolerance` and smaller `num_anchor_steps` give faster sampling; smaller `tolerance` and
larger `num_anchor_steps` give higher fidelity to the sequential trajectory.

## Repository Structure

```
ParaFlow/
├── paraflow/
│   ├── __init__.py
│   ├── pipeline_flux.py        # ParaFlowFluxPipeline  (FLUX.1)
│   └── pipeline_sd3.py         # ParaFlowSD3Pipeline   (SD3 / SD3.5)
├── generate_paraflow.py        # ParaFlow generation / benchmark script
├── generate_sequential.py      # Sequential baselines (Euler / UniPC / DPM-Solver)
├── data/
│   └── prompts.txt             # 10k evaluation prompts
├── requirements.txt
└── LICENSE
```

## Citation

If you find ParaFlow useful in your research, please cite:

```bibtex
@inproceedings{lu2026paraflow,
  title     = {ParaFlow: Parallel Sampling for Flow Matching Models},
  author    = {Lu, Jianrong and Li, Bangwei and Zhang, Haomin and Gu, Zhuoya and Lu, Yongqing and Chen, Jianhai and He, Qinming},
  booktitle = {Computer Vision -- ECCV 2026},
  series    = {Lecture Notes in Computer Science},
  pages     = {128--145},
  year      = {2026},
  publisher = {Springer Nature Switzerland},
  doi       = {10.1007/978-3-032-37038-9_8}
}
```

## Acknowledgements

Our implementation builds on [Hugging Face diffusers](https://github.com/huggingface/diffusers).
The sliding-window parallel sampling scheme is inspired by
[ParaDiGMS](https://github.com/prowlive/paradigms) (Shih et al., *Parallel Sampling of Diffusion
Models*, NeurIPS 2023). We thank the authors for their excellent open-source work.

## License

This project is released under the [Apache License 2.0](LICENSE).
