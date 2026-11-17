# TAARHOGEN2 Architecture

## Non-negotiable project boundary

TAARHOGEN2 is a new image-generation project. It does not import, load, convert, or depend on TAARHOGEN1, CLIP ViT-B/32, its projection layer, its embeddings, or prior decoder/bridge experiments.

Only the Studio's user-facing creative controls are retained. The view names and photographic settings are product configuration, not inherited model architecture.

## Runtime layers

1. **Studio controls** — validated creative inputs (view, focal length, aperture, lighting, environment, orbit, scale, finish, quality).
2. **Prompt conditioning** — translates descriptive Studio intent into prompt text. It does not claim to simulate physical optics.
3. **Model adapter** — loads the selected complete text-to-image model through a local runtime.
4. **Generation** — executes the model's native inference pipeline.
5. **Native decode** — uses the model pipeline's matched latent decoder/autoencoder; no CLIP-to-decoder bridge.
6. **Output validation** — verifies the generated PNG and writes a JSON sidecar with prompt, seed, dimensions, model ID and Studio settings.

## Initial model adapter

The first integration target is FLUX.2 Klein 4B through Diffusers. Model ID is configurable using TAARHOGEN2_MODEL_ID. The upstream model weights are downloaded at runtime and are not committed here.

The model selection remains subject to runtime validation. Measure GPU type, VRAM, RAM, disk and model-load success in the target Colab session before declaring the configuration supported.

## Studio controls and model semantics

The first scaffold maps lens/lighting/finish values into natural-language prompt guidance. These are creative controls, not calibrated physical camera simulation. Orbit and scene scale are retained as Studio state for future scene-aware rendering; they do not yet transform a 3D scene in this text-to-image adapter.

## Acceptance gate: first PNG

- Runtime reports its actual hardware.
- Model loads locally without a paid API.
- One image is generated from a text prompt.
- File opens as PNG and passes image verification.
- Dimensions and deterministic seed are recorded in a sidecar JSON.
- Exact runtime logs and peak memory are recorded for reproducibility.
