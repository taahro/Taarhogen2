"""Runtime configuration for the independent TAARHOGEN2 engine."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class EngineConfig:
    model_id: str = os.getenv(
        "TAARHOGEN2_MODEL_ID",
        "black-forest-labs/FLUX.2-klein-4B",
    )
    output_dir: str = os.getenv("TAARHOGEN2_OUTPUT_DIR", "outputs")
    default_width: int = int(os.getenv("TAARHOGEN2_WIDTH", "1024"))
    default_height: int = int(os.getenv("TAARHOGEN2_HEIGHT", "1024"))
    default_steps: int = int(os.getenv("TAARHOGEN2_STEPS", "4"))
    default_guidance: float = float(os.getenv("TAARHOGEN2_GUIDANCE", "1.0"))


CONFIG = EngineConfig()
