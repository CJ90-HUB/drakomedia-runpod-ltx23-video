from __future__ import annotations

from collections.abc import Iterator

import torch

from ltx_core.components.noisers import GaussianNoiser
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_core.model.video_vae import TilingConfig
from ltx_core.quantization import QuantizationPolicy
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.blocks import (
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
    VideoUpsampler,
)
from ltx_pipelines.utils.constants import (
    DISTILLED_SIGMAS,
    STAGE_2_DISTILLED_SIGMAS,
)
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    combined_image_conditionings,
    get_device,
)
from ltx_pipelines.utils.types import ModalitySpec, OffloadMode


class VideoOnlyDistilledPipeline(DistilledPipeline):
    """Two-stage LTX-2.3 pipeline that never creates audio latents."""

    def __init__(
        self,
        distilled_checkpoint_path: str,
        gemma_root: str,
        spatial_upsampler_path: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device | None = None,
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        compilation_config: CompilationConfig | None = None,
        offload_mode: OffloadMode = OffloadMode.NONE,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
    ):
        self.device = device or get_device()
        self.dtype = torch.bfloat16
        self.prompt_encoder = PromptEncoder(
            distilled_checkpoint_path,
            gemma_root,
            self.dtype,
            self.device,
            registry=registry,
            offload_mode=offload_mode,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.image_conditioner = ImageConditioner(
            distilled_checkpoint_path,
            self.dtype,
            self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.stage = DiffusionStage.from_checkpoint(
            distilled_checkpoint_path,
            self.dtype,
            self.device,
            loras=tuple(loras),
            quantization=quantization,
            registry=registry,
            compilation_config=compilation_config,
            offload_mode=offload_mode,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.upsampler = VideoUpsampler(
            distilled_checkpoint_path,
            spatial_upsampler_path,
            self.dtype,
            self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
        )
        self.video_decoder = VideoDecoder(
            distilled_checkpoint_path,
            self.dtype,
            self.device,
            registry=registry,
            alloc_trim_strategy=alloc_trim_strategy,
        )

    @torch.inference_mode()
    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        stage_1_sigmas: torch.Tensor = DISTILLED_SIGMAS,
        stage_2_sigmas: torch.Tensor = STAGE_2_DISTILLED_SIGMAS,
    ) -> Iterator[torch.Tensor]:
        assert_resolution(height=height, width=width, is_two_stage=True)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        dtype = torch.bfloat16

        (context,) = self.prompt_encoder(
            [prompt],
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_image=images[0][0] if images else None,
        )
        video_context = context.video_encoding

        stage_1_sigmas = stage_1_sigmas.to(
            dtype=torch.float32,
            device=self.device,
        )
        stage_1_width, stage_1_height = width // 2, height // 2
        stage_1_conditionings = self.image_conditioner(
            lambda encoder: combined_image_conditionings(
                images=images,
                height=stage_1_height,
                width=stage_1_width,
                video_encoder=encoder,
                dtype=dtype,
                device=self.device,
            )
        )
        video_state, _ = self.stage(
            denoiser=SimpleDenoiser(video_context, None),
            sigmas=stage_1_sigmas,
            noiser=noiser,
            width=stage_1_width,
            height=stage_1_height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=video_context,
                conditionings=stage_1_conditionings,
            ),
        )
        if video_state is None:
            raise RuntimeError("LTX-2.3 no produjo latentes de vídeo.")

        upscaled_video_latent = self.upsampler(video_state.latent[:1])
        stage_2_sigmas = stage_2_sigmas.to(
            dtype=torch.float32,
            device=self.device,
        )
        stage_2_conditionings = self.image_conditioner(
            lambda encoder: combined_image_conditionings(
                images=images,
                height=height,
                width=width,
                video_encoder=encoder,
                dtype=dtype,
                device=self.device,
            )
        )
        video_state, _ = self.stage(
            denoiser=SimpleDenoiser(video_context, None),
            sigmas=stage_2_sigmas,
            noiser=noiser,
            width=width,
            height=height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=video_context,
                conditionings=stage_2_conditionings,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=upscaled_video_latent,
            ),
        )
        if video_state is None:
            raise RuntimeError("LTX-2.3 no refinó los latentes de vídeo.")
        return self.video_decoder(
            video_state.latent,
            tiling_config,
            generator,
        )
