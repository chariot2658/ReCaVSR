# ReCaVSR

Official inference code for [ReCaVSR: One-Step Streaming Diffusion Video
Super-Resolution with Recycled Latents and Learned Cache Routing](https://arxiv.org/abs/2609.37831).

Authors: Xijun Wang, Xin Li, Suhang Yao, Zirui Lang, Bingchen Li, and Zhibo Chen.

## Installation

Linux x86-64, Python 3.11–3.13, and an NVIDIA GPU with a CUDA 13.0-compatible driver
are required. Install [uv](https://docs.astral.sh/uv/getting-started/installation/),
then run from the repository root:

```bash
uv sync --locked
```

## Pretrained Models

Model repository: [anonyaa/ReCaVSR](https://huggingface.co/anonyaa/ReCaVSR)
on Hugging Face. Checkpoint files are downloaded separately and are not included
in this code repository.

The three JSON configuration files are included in this code repository under
`checkpoints/`; they do not need to be downloaded from Hugging Face.
Download only the weights **before running inference**:

```bash
uvx hf download anonyaa/ReCaVSR \
  transformer.safetensors prompt.safetensors flashdecoder.safetensors \
  --local-dir checkpoints
```


```text
checkpoints/
├── transformer.safetensors
├── model_config.json        
├── prompt.safetensors
├── vae_config.json            
├── flashdecoder.safetensors
└── flashdecoder_config.json 
```

Use the bundled JSON files that match the checkpoint release. If you use a
different `--model-dir`, copy these three JSON files into that directory as well.
The DiT checkpoint is already merged; no conversion or merging is required.

## Quick Start

After downloading the pretrained models:

```bash
uv run python inference.py \
  --input assets/demo/input.mp4 \
  --output outputs/demo_x4.mp4 \
  --scale 4
```

Useful options:

- `--scale 2` or `--scale 1.5`: change the spatial upscaling factor.
- `--device cuda:1`: select a GPU.
- `--color-fix wavelet` or `--color-fix adain`: enable color correction.
- `--model-dir /path/to/checkpoints`: use a different download directory.
- `--frames 31`: run a short preview before processing the full demo.

Compilation is enabled by default, so the first run takes longer.
Use `--no-compile-blocks --no-compile-decoder` to disable it.
Input may also be a numbered image directory; use `--fps` to set its playback rate.
See `uv run python inference.py --help` for all options.

### Optional Wan Decoder

To use the original Wan decoder instead of FlashDecoder, also download:

```bash
uvx hf download anonyaa/ReCaVSR vae.safetensors --local-dir checkpoints
```

Then add `--decoder wan` to the inference command.

Model computation is streaming; the current video reader and writer buffer the
clip in CPU memory. Outputs preserve frame count, contain no audio, and do not
overwrite existing files.

## Acknowledgements

This project builds on Wan and Hugging Face Diffusers. Color correction follows
[StableSR](https://github.com/IceClear/StableSR). Upstream copyright notices are
retained in the source files.

## License

The code is released under the Apache License 2.0.
Model weights and demo footage remain subject to their respective licenses.

## Branches

`main` is the public release. `anonymous` preserves the anonymous code snapshot.
