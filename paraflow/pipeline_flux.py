import torch
import inspect
import numpy as np

from typing import Any, Callable, Dict, List, Optional, Union
from diffusers import FluxPipeline
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput

from diffusers.utils import (
    is_torch_xla_available,
    logging,
    replace_example_docstring,
)

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from paraflow import ParaFlowFluxPipeline

        >>> pipe = ParaFlowFluxPipeline.from_pretrained(
        ...     "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
        ... )
        >>> pipe.to("cuda")

        >>> prompt = "A cat holding a sign that says hello world"
        >>> output, num_iterations = pipe(
        ...     prompt,
        ...     height=1024,
        ...     width=1024,
        ...     guidance_scale=3.5,
        ...     num_inference_steps=50,
        ...     window_size=50,
        ...     num_anchor_steps=1,
        ...     tolerance=0.01,
        ...     max_iterations=15,
        ...     max_sequence_length=512,
        ...     generator=torch.Generator("cpu").manual_seed(0),
        ... )
        >>> output.images[0].save("paraflow_flux.png")
        ```
"""


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class ParaFlowFluxPipeline(FluxPipeline):
    """FLUX text-to-image pipeline accelerated with ParaFlow parallel sampling.

    ParaFlow (ECCV 2026) speeds up flow-matching sampling with a sliding window over the
    denoising timesteps. In every iteration, only the first `num_anchor_steps` steps of the
    window are evaluated with the transformer; the remaining steps of the window are
    approximated by a first-order Taylor expansion of the velocity field around the last
    anchor step. The window then advances past all steps whose accumulated error stays below
    `tolerance`.
    """

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        window_size: int = 10,
        max_iterations: int = 50,
        num_anchor_steps: int = 50,
        tolerance: float = 0.01,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[PipelineImageInput] = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        max_sequence_length: int = 512,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
                not greater than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in all the text-encoders.
            true_cfg_scale (`float`, *optional*, defaults to 1.0):
                When > 1.0 and a provided `negative_prompt`, enables true classifier-free guidance.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            window_size (`int`, *optional*, defaults to 10):
                Size of the sliding window of timesteps processed in each ParaFlow iteration.
            num_anchor_steps (`int`, *optional*):
                Number of leading steps in each window that are evaluated with the transformer. The
                remaining steps of the window are approximated by a first-order Taylor expansion of the
                velocity field around the last anchor step.
            tolerance (`float`, *optional*, defaults to 0.01):
                Relative error tolerance. After each iteration the window advances past all steps whose
                accumulated error stays below this threshold.
            max_iterations (`int`, *optional*, defaults to 50):
                Maximum number of ParaFlow iterations.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 3.5):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            negative_ip_adapter_image:
                (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            negative_ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            `tuple`: A pair `(output, num_iterations)`, where `output` is a
            [`~pipelines.flux.FluxPipelineOutput`] (or a plain tuple if `return_dict=False`) and
            `num_iterations` is the number of ParaFlow iterations used.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            # callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        if do_true_cfg:
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                negative_text_ids,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
            negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None
        ):
            negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            negative_ip_adapter_image = [negative_ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

        elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
            negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None
        ):
            ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            ip_adapter_image = [ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        image_embeds = None
        negative_image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )
        if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
            negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                negative_ip_adapter_image,
                negative_ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )

        # 6. Denoising loop
        # We set the index here to remove DtoH sync, helpful especially during compilation.
        # Check out more details here: https://github.com/huggingface/diffusers/pull/11696
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            num_iterations = 0
            window_size = min(window_size, num_inference_steps)
            window_start = 0
            window_end = window_size
            scaled_tolerance = tolerance**2
            latents_time_evolution_buffer = torch.stack([latents] * (num_inference_steps + 1))

            all_timesteps = torch.cat((timesteps, torch.zeros(1).to(timesteps.device)), 0)

            num_anchor_steps = min(num_anchor_steps, window_size)
            prev_anchor_velocity = None
            anchor_velocity = None

            anchor_timestep = None
            prev_anchor_timestep = None
            while window_start < num_inference_steps and num_iterations < max_iterations:
                if self.interrupt:
                    continue

                self._current_timestep = all_timesteps[window_start]
                if image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds

                window_len = window_end - window_start
                num_anchor_steps = min(num_anchor_steps, window_len)

                window_timesteps = all_timesteps[window_start:window_end]

                window_velocities = None
                for i in range(window_start, window_end):
                    latents = latents_time_evolution_buffer[i]
                    t = timesteps[i]
                    timestep = t.expand(latents.shape[0]).to(latents.dtype)
                    if i < window_start + num_anchor_steps:
                        with self.transformer.cache_context("cond"):
                            velocity = self.transformer(
                                hidden_states=latents,
                                timestep=timestep / 1000,
                                guidance=guidance,
                                pooled_projections=pooled_prompt_embeds,
                                encoder_hidden_states=prompt_embeds,
                                txt_ids=text_ids,
                                img_ids=latent_image_ids,
                                joint_attention_kwargs=self.joint_attention_kwargs,
                                return_dict=False,
                            )[0]
                        if do_true_cfg:
                            if negative_image_embeds is not None:
                                self._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds
                            with self.transformer.cache_context("uncond"):
                                neg_velocity = self.transformer(
                                    hidden_states=latents,
                                    timestep=timestep / 1000,
                                    guidance=guidance,
                                    pooled_projections=negative_pooled_prompt_embeds,
                                    encoder_hidden_states=negative_prompt_embeds,
                                    txt_ids=negative_text_ids,
                                    img_ids=latent_image_ids,
                                    joint_attention_kwargs=self.joint_attention_kwargs,
                                    return_dict=False,
                                )[0]
                            velocity = neg_velocity + true_cfg_scale * (velocity - neg_velocity)

                    if num_iterations == 0:
                        if i == window_start + num_anchor_steps - 1:
                            prev_anchor_velocity = velocity.clone()
                            prev_anchor_timestep = t
                    else:
                        if i <= window_start + num_anchor_steps - 1:
                            anchor_velocity = velocity.clone()
                            anchor_timestep = t
                        else:
                            velocity_slope = (anchor_velocity - prev_anchor_velocity) / (
                                anchor_timestep - prev_anchor_timestep
                            )
                            dt_from_anchor = t - anchor_timestep
                            velocity = anchor_velocity + velocity_slope * dt_from_anchor
                            if i == window_end - 1:
                                prev_anchor_velocity = anchor_velocity.clone()
                                prev_anchor_timestep = anchor_timestep

                    if window_velocities is None:
                        window_velocities = velocity
                    else:
                        window_velocities = torch.cat((window_velocities, velocity), 0)

                window_timesteps = window_timesteps.to(window_velocities.device)
                next_window_timesteps = all_timesteps[window_start + 1 : window_end + 1].to(window_velocities.device)

                # back to shape (parallel_dim, batch_size, ...)
                # now we want to add the pre-sampled noise
                # parallel sampling algorithm requires computing the cumulative drift from the beginning
                # of the window, so we need to compute cumulative sum of the deltas and the pre-sampled noises.
                delta = window_velocities * (next_window_timesteps - window_timesteps)[:, None, None] / 1000
                cumulative_delta = torch.cumsum(delta, dim=0)

                new_window_latents = latents_time_evolution_buffer[window_start][None,] + cumulative_delta[:, None]

                cur_error_vec = (
                    new_window_latents - latents_time_evolution_buffer[window_start + 1 : window_end + 1]
                ).reshape(window_len, new_window_latents.shape[1], -1)
                cur_error = torch.linalg.norm(cur_error_vec, dim=-1).pow(2)
                error_ratio = cur_error / latents.numel()

                # find the first index of the vector error_ratio that is greater than error tolerance
                # we can shift the window for the next iteration up to this index
                error_ratio = torch.nn.functional.pad(
                    error_ratio, (0, 0, 0, 1), value=1e9
                )  # handle the case when everything is below ratio, by padding the end of window_len dimension
                any_error_at_time = torch.max(error_ratio > scaled_tolerance, dim=1).values.int()
                first_error_idx = torch.argmax(any_error_at_time).item()

                # compute the new begin and end idxs for the window
                new_window_start = window_start + min(1 + first_error_idx, window_size)
                new_window_end = min(new_window_start + window_size, num_inference_steps)

                # store the computed latents for the current window in the global buffer
                latents_time_evolution_buffer[window_start + 1 : window_end + 1] = new_window_latents
                # initialize the new sliding window latents with the end of the current window,
                # should be better than random initialization
                latents_time_evolution_buffer[window_end : new_window_end + 1] = latents_time_evolution_buffer[
                    window_end
                ][
                    None,
                ]

                num_iterations += 1

                # call the callback, if provided
                progress_bar.update(new_window_start - window_start)
                if callback is not None and num_iterations % callback_steps == 0:
                    callback(window_start, window_timesteps[window_start], latents_time_evolution_buffer[window_start])

                if XLA_AVAILABLE:
                    xm.mark_step()

                window_start = new_window_start
                window_end = new_window_end

        logger.info(f"ParaFlow: finished in {num_iterations} parallel iterations.")

        latents = latents_time_evolution_buffer[-1]
        del latents_time_evolution_buffer

        self._current_timestep = None

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,), num_iterations

        return FluxPipelineOutput(images=image), num_iterations
