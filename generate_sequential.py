"""Sequential (baseline) image generation with standard diffusers schedulers.

Supports the flow-matching Euler, UniPC and DPM-Solver schedulers. Useful for
reproducing the sequential baselines reported in the ParaFlow paper.

Example:
    python generate_sequential.py --model flux --sampler euler --num_inference_steps 50
"""

import argparse
import os
import time

import torch
from diffusers import (
    DPMSolverMultistepScheduler,
    FluxPipeline,
    FlowMatchEulerDiscreteScheduler,
    StableDiffusion3Pipeline,
    UniPCMultistepScheduler,
)

# model name -> (Hugging Face repo id, pipeline class, dtype, default guidance_scale)
MODEL_REGISTRY = {
    "flux": ("black-forest-labs/FLUX.1-dev", FluxPipeline, torch.bfloat16, 3.5),
    "sd3": ("stabilityai/stable-diffusion-3-medium-diffusers", StableDiffusion3Pipeline, torch.float16, 5.0),
    "sd3.5": ("stabilityai/stable-diffusion-3.5-large", StableDiffusion3Pipeline, torch.bfloat16, 5.0),
}


def apply_sampler(pipe, sampler_name: str):
    """Swap the scheduler of `pipe` for the requested flow-matching sampler."""
    if sampler_name == "euler":
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    elif sampler_name == "unipc":
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config,
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            solver_order=2,
            lower_order_final=True,
        )
    elif sampler_name == "dpm":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            algorithm_type="dpmsolver++",
            solver_order=2,
            solver_type="midpoint",
            lower_order_final=True,
        )
    else:
        raise ValueError(f"Unknown sampler: {sampler_name}. Choose from: euler | unipc | dpm")
    return pipe


def parse_args():
    parser = argparse.ArgumentParser(description="Sequential generation with standard diffusers schedulers.")
    parser.add_argument("--model", type=str, default="flux", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--sampler", type=str, default="euler", choices=["euler", "unipc", "dpm"])
    parser.add_argument("--prompt_file", type=str, default="data/prompts.txt")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--num_images", type=int, default=5000)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--guidance_scale", type=float, default=None, help="Defaults to the per-model value.")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--cpu_offload", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    model_id, pipeline_cls, dtype, default_guidance_scale = MODEL_REGISTRY[args.model]
    guidance_scale = args.guidance_scale if args.guidance_scale is not None else default_guidance_scale

    pipe = pipeline_cls.from_pretrained(model_id, torch_dtype=dtype, local_files_only=args.local_files_only)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(f"cuda:{args.gpu_id}")
    pipe = apply_sampler(pipe, args.sampler)

    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    start_index = args.start_index
    end_index = min(start_index + args.num_images, len(prompts))
    if start_index >= end_index:
        raise ValueError(f"start_index={start_index} is out of range for {len(prompts)} prompts.")

    timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
    run_name = (
        f"sequential_{args.model}_{args.sampler}_{args.width}x{args.height}"
        f"_{args.num_inference_steps}steps_{timestamp}"
    )
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    generate_kwargs = {
        "height": args.height,
        "width": args.width,
        "guidance_scale": guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "generator": torch.Generator("cpu").manual_seed(args.seed),
    }
    if args.model == "flux":
        generate_kwargs["max_sequence_length"] = 512

    start_time = time.time()
    for idx in range(start_index, end_index):
        image = pipe(prompts[idx], **generate_kwargs).images[0]
        image.save(os.path.join(run_dir, f"{idx:04d}.png"))

    num_generated = end_index - start_index
    total_time = time.time() - start_time
    mean_time = total_time / num_generated
    print(f"Generated {num_generated} images in {total_time:.1f}s ({mean_time:.2f}s/image).")

    with open(os.path.join(run_dir, "result_log.txt"), "w") as log_file:
        log_file.write(f"Model: {args.model} ({model_id})\n")
        log_file.write(f"Sampler: {args.sampler} ({pipe.scheduler.__class__.__name__})\n")
        log_file.write(f"Config: {vars(args)}\n")
        log_file.write(f"Generated {num_generated} images.\n")
        log_file.write(f"Mean time per image: {mean_time:.2f}s\n")
        log_file.write(f"Total time: {total_time:.1f}s\n")


if __name__ == "__main__":
    main()
