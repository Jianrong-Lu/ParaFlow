"""Generate images with ParaFlow parallel sampling.

Example:
    python generate_paraflow.py --model flux --num_images 5000
"""

import argparse
import os
import time

import torch

from paraflow import ParaFlowFluxPipeline, ParaFlowSD3Pipeline

# model name -> (Hugging Face repo id, pipeline class, dtype, default max_sequence_length)
MODEL_REGISTRY = {
    "flux": ("black-forest-labs/FLUX.1-dev", ParaFlowFluxPipeline, torch.bfloat16, 512),
    "sd3": ("stabilityai/stable-diffusion-3-medium-diffusers", ParaFlowSD3Pipeline, torch.float16, 256),
    "sd3.5": ("stabilityai/stable-diffusion-3.5-large", ParaFlowSD3Pipeline, torch.bfloat16, 256),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images with ParaFlow parallel sampling.")
    parser.add_argument("--model", type=str, default="flux", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--prompt_file", type=str, default="data/prompts.txt")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--start_index", type=int, default=0, help="First prompt index to generate.")
    parser.add_argument("--num_images", type=int, default=5000, help="Number of images to generate.")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--window_size", type=int, default=50, help="Sliding-window size.")
    parser.add_argument(
        "--num_anchor_steps",
        type=int,
        default=1,
        help="Leading window steps evaluated with the transformer; the rest are extrapolated.",
    )
    parser.add_argument("--tolerance", type=float, default=0.01, help="Error tolerance for advancing the window.")
    parser.add_argument("--max_iterations", type=int, default=15, help="Maximum number of parallel iterations.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--local_files_only", action="store_true", help="Load models from the local HF cache only.")
    parser.add_argument("--cpu_offload", action="store_true", help="Enable model CPU offload to save VRAM.")
    return parser.parse_args()


def main():
    args = parse_args()
    model_id, pipeline_cls, dtype, max_sequence_length = MODEL_REGISTRY[args.model]

    pipe = pipeline_cls.from_pretrained(model_id, torch_dtype=dtype, local_files_only=args.local_files_only)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(f"cuda:{args.gpu_id}")

    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    start_index = args.start_index
    end_index = min(start_index + args.num_images, len(prompts))
    if start_index >= end_index:
        raise ValueError(f"start_index={start_index} is out of range for {len(prompts)} prompts.")

    timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
    run_name = (
        f"paraflow_{args.model}_tol{args.tolerance}_w{args.window_size}_a{args.num_anchor_steps}"
        f"_{args.width}x{args.height}_{args.num_inference_steps}steps_{timestamp}"
    )
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    total_iterations = 0
    start_time = time.time()
    for idx in range(start_index, end_index):
        output, num_iterations = pipe(
            prompts[idx],
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            window_size=args.window_size,
            num_anchor_steps=args.num_anchor_steps,
            tolerance=args.tolerance,
            max_iterations=args.max_iterations,
            max_sequence_length=max_sequence_length,
            generator=torch.Generator("cpu").manual_seed(args.seed),
        )
        output.images[0].save(os.path.join(run_dir, f"{idx:04d}.png"))
        total_iterations += num_iterations

    num_generated = end_index - start_index
    total_time = time.time() - start_time
    mean_iterations = total_iterations / num_generated
    mean_time = total_time / num_generated
    print(f"Generated {num_generated} images in {total_time:.1f}s ({mean_time:.2f}s/image).")
    print(f"Mean parallel iterations: {mean_iterations:.2f}")

    with open(os.path.join(run_dir, "result_log.txt"), "w") as log_file:
        log_file.write(f"Model: {args.model} ({model_id})\n")
        log_file.write(f"Config: {vars(args)}\n")
        log_file.write(f"Generated {num_generated} images.\n")
        log_file.write(f"Mean parallel iterations: {mean_iterations:.2f}\n")
        log_file.write(f"Mean time per image: {mean_time:.2f}s\n")
        log_file.write(f"Total time: {total_time:.1f}s\n")


if __name__ == "__main__":
    main()
