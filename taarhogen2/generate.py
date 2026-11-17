"""Text-to-image generation using the FLUX.2 Klein Diffusers pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from .config import CONFIG
from .studio import StudioSettings, studio_prompt_suffix

LOG = logging.getLogger("taarhogen2")


def _runtime_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_pipeline(model_id: Optional[str] = None):
    """Load the model locally. No hosted inference API is called."""
    try:
        from diffusers import Flux2KleinPipeline
    except ImportError as exc:
        raise RuntimeError(
            "Flux2KleinPipeline is unavailable. Install the current Diffusers "
            "package from requirements.txt."
        ) from exc

    dtype = _runtime_dtype()
    LOG.info("Loading %s using dtype=%s", model_id or CONFIG.model_id, dtype)
    pipe = Flux2KleinPipeline.from_pretrained(
        model_id or CONFIG.model_id,
        torch_dtype=dtype,
    )

    if torch.cuda.is_available():
        # Sequential component offload reduces VRAM pressure at the cost of speed.
        pipe.enable_model_cpu_offload()
    else:
        LOG.warning(
            "CUDA is unavailable. CPU generation may be extremely slow or "
            "impractical for this model."
        )
        pipe.to("cpu")
    return pipe


def validate_png(path: Path) -> tuple[int, int]:
    """Verify the output is a readable PNG and return its dimensions."""
    with Image.open(path) as image:
        if image.format != "PNG":
            raise ValueError(f"Expected PNG output, received {image.format}")
        image.verify()
    with Image.open(path) as image:
        return image.size


def generate_image(
    prompt: str,
    output: str | Path,
    *,
    width: int = CONFIG.default_width,
    height: int = CONFIG.default_height,
    steps: int = CONFIG.default_steps,
    guidance: float = CONFIG.default_guidance,
    seed: int = 424242,
    settings: Optional[StudioSettings] = None,
    model_id: Optional[str] = None,
) -> Path:
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if width < 256 or height < 256 or width % 16 or height % 16:
        raise ValueError("width and height must be >=256 and divisible by 16")
    if steps < 1:
        raise ValueError("steps must be positive")

    settings = settings or StudioSettings()
    settings.validate()
    final_prompt = f"{prompt.strip()}\n\n{studio_prompt_suffix(settings)}"

    pipe = load_pipeline(model_id)
    generator_device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(seed)

    result = pipe(
        prompt=final_prompt,
        height=height,
        width=width,
        guidance_scale=guidance,
        num_inference_steps=steps,
        generator=generator,
    )
    image = result.images[0]

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")
    dimensions = validate_png(output_path)

    metadata = {
        "engine": "TAARHOGEN2",
        "model_id": model_id or CONFIG.model_id,
        "prompt": prompt,
        "conditioned_prompt": final_prompt,
        "seed": seed,
        "width": dimensions[0],
        "height": dimensions[1],
        "steps": steps,
        "guidance": guidance,
        "studio_settings": settings.to_dict(),
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    LOG.info("Verified PNG written to %s (%sx%s)", output_path, *dimensions)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an image with TAARHOGEN2")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default=f"{CONFIG.output_dir}/taarhogen2.png")
    parser.add_argument("--width", type=int, default=CONFIG.default_width)
    parser.add_argument("--height", type=int, default=CONFIG.default_height)
    parser.add_argument("--steps", type=int, default=CONFIG.default_steps)
    parser.add_argument("--guidance", type=float, default=CONFIG.default_guidance)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--view", choices=[f"VIEW 0{i}" for i in range(1, 6)], default="VIEW 01")
    parser.add_argument("--focal-length", type=int, default=50)
    parser.add_argument("--aperture", type=float, default=2.8)
    parser.add_argument("--lighting", default="soft studio")
    parser.add_argument("--environment", default="studio")
    parser.add_argument("--finish", default="natural")
    parser.add_argument("--model-id", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    settings = StudioSettings(
        view=args.view,
        focal_length_mm=args.focal_length,
        aperture=args.aperture,
        lighting=args.lighting,
        environment=args.environment,
        finish=args.finish,
    )
    path = generate_image(
        args.prompt,
        args.output,
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
        settings=settings,
        model_id=args.model_id,
    )
    print(f"PNG_READY={path}")


if __name__ == "__main__":
    main()
