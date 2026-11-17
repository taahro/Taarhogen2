from taarhogen2.config import CONFIG


def test_default_model_is_complete_generation_pipeline():
    assert CONFIG.model_id == "black-forest-labs/FLUX.2-klein-4B"


def test_default_image_dimensions_are_supported():
    assert CONFIG.default_width >= 256
    assert CONFIG.default_height >= 256
    assert CONFIG.default_width % 16 == 0
    assert CONFIG.default_height % 16 == 0
