import pytest

from taarhogen2.studio import StudioSettings, studio_prompt_suffix


def test_default_studio_settings_are_valid():
    settings = StudioSettings()
    settings.validate()
    assert settings.view == "VIEW 01"


def test_view_range_is_enforced():
    with pytest.raises(ValueError):
        StudioSettings(view="VIEW 06").validate()


def test_studio_controls_become_prompt_guidance():
    settings = StudioSettings(focal_length_mm=85, aperture=1.8)
    suffix = studio_prompt_suffix(settings)
    assert "85mm" in suffix
    assert "f/1.8" in suffix
