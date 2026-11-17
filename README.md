# TAARHOGEN2

**TAARHO Independent Image Generation Engine**

A clean image-generation project with the TAARHO Virtual Photography Studio controls as its product interface. This repository does not depend on TAARHOGEN1, its CLIP encoder, embeddings, weights, decoder experiments, or any previous model bridge.

## Project boundary

### Carried forward from the Studio
- Five public view profiles: VIEW 01–VIEW 05
- Focal length and aperture controls
- Lighting and environment controls
- Camera orbit and scene scale
- Finish and quality settings
- PNG export

These are interface and capture controls. They are not model weights or inherited neural architecture.

### Fresh engine
- Independent prompt-to-image pipeline
- Explicit model adapter and runtime configuration
- Latent generation and model-native decoding
- PNG validation and reproducible generation metadata
- Hardware-aware execution for free Colab/local GPU testing

## Foundation candidate

Initial integration target: `black-forest-labs/FLUX.2-klein-4B` through the Diffusers `Flux2KleinPipeline`.

The model is Apache-2.0 licensed according to its official model card. The upstream model weights are not included in this repository and remain subject to their own license and usage terms. Hardware requirements must be measured on the actual runtime; the official model card describes approximately 13 GB VRAM, so a free Colab GPU is not assumed to be sufficient.

Official references:
- [FLUX.2 inference repository](https://github.com/black-forest-labs/flux2)
- [FLUX.2 Klein 4B model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)

No hosted inference API is used by this project. The first target is local/free Colab execution, subject to available hardware.

## Repository layout

```
taarhogen2/
  __init__.py
  config.py
  studio.py
  generate.py
scripts/
  generate.py
notebooks/
  TAARHOGEN2_First_Generation.ipynb
tests/
  test_studio.py
requirements.txt
```

## Quick start

Use Python 3.11 or 3.12 with a compatible PyTorch/CUDA runtime.

```bash
python -m pip install -r requirements.txt
python scripts/generate.py --prompt "A cinematic studio portrait of a silver vintage sports car" --output outputs/first-generation.png
```

The first run downloads model components from Hugging Face. This can require substantial disk space, RAM, VRAM and network bandwidth. No token is required for the public model unless upstream access rules change.

## Security

- Never commit API keys, Hugging Face tokens, passwords, or personal credentials.
- Model weights and generated outputs are excluded from Git.
- Do not place private camera profile data or customer images in this public repository.

## Status

Initial architecture scaffold. Generation is not considered verified until a runtime successfully loads the model, generates an image, and passes PNG validation.
