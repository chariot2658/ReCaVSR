"""Fixed geometry of the released checkpoint, not tunable chunk sizes."""

# The causal VAE emits 1 + 4 * (T - 1) RGB frames for the first latent block,
# and four RGB frames per latent thereafter.
PREFIX_LATENT_FRAMES = 6
BODY_LATENT_FRAMES = 2
PREFIX_RGB_FRAMES = 21
BODY_RGB_FRAMES = 8

VAE_SPATIAL_STRIDE = 16
SPATIAL_TOKEN_STRIDE = 32  # VAE stride 16 followed by a 2x2 DiT patch.
ROPE_WINDOW_LATENTS = 22
INFERENCE_TIMESTEP = 1000.0
DEFAULT_SPATIAL_WINDOW = (22, 40)  # Token units, not RGB pixels.
