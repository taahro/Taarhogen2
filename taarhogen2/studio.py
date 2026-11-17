"""Studio controls retained as product settings, not inherited model architecture.

The view identifiers are deliberately generic. Camera brand/profile internals are
not exposed through the public interface.
"""

from dataclasses import asdict, dataclass
from typing import Literal


ViewId = Literal["VIEW 01", "VIEW 02", "VIEW 03", "VIEW 04", "VIEW 05"]


@dataclass
class StudioSettings:
    view: ViewId = "VIEW 01"
    focal_length_mm: int = 50
    aperture: float = 2.8
    lighting: str = "soft studio"
    environment: str = "studio"
    orbit_degrees: float = 0.0
    scene_scale: float = 1.0
    finish: str = "natural"
    quality: str = "high"

    def validate(self) -> None:
        if self.view not in {f"VIEW 0{i}" for i in range(1, 6)}:
            raise ValueError("view must be VIEW 01 through VIEW 05")
        if not 10 <= self.focal_length_mm <= 300:
            raise ValueError("focal_length_mm must be between 10 and 300")
        if not 0.7 <= self.aperture <= 32:
            raise ValueError("aperture must be between f/0.7 and f/32")
        if not -360 <= self.orbit_degrees <= 360:
            raise ValueError("orbit_degrees must be between -360 and 360")
        if not 0.1 <= self.scene_scale <= 10:
            raise ValueError("scene_scale must be between 0.1 and 10")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)


def studio_prompt_suffix(settings: StudioSettings) -> str:
    """Translate selected creative intent into natural-language prompt guidance.

    This is prompt conditioning only; it is not a physical camera simulator.
    """
    settings.validate()
    return (
        f"Photographic direction: {settings.focal_length_mm}mm lens perspective, "
        f"f/{settings.aperture:g} depth-of-field character, "
        f"{settings.lighting} lighting, {settings.environment} environment, "
        f"{settings.finish} finish, high-detail {settings.quality} image."
    )
