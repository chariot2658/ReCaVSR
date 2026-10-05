# ReCaVSR

**ReCaVSR: One-Step Streaming Diffusion Video Super-Resolution with Recycled Latents and Learned Cache Routing**

Xijun Wang, Xin Li, Suhang Yao, Zirui Lang, Bingchen Li, and Zhibo Chen

[Paper](https://arxiv.org/abs/2609.37831) · [Pretrained models](https://huggingface.co/kopper/ReCaVSR) · [Code](https://github.com/kopperx/ReCaVSR)

ReCaVSR is a one-step streaming video super-resolution method built on Wan2.2. It
reuses previously generated super-resolution latents, assigns different temporal
cache scopes to transformer layers, and decodes with a low-resolution-conditioned
FlashDecoder. This repository provides the inference code and links to pretrained
weights. Training and evaluation pipelines are not included in this release.

## Visual Results

| Low-resolution input | ReCaVSR output (4×) |
| :---: | :---: |
| https://github.com/user-attachments/assets/e78d3868-b06e-4a57-b025-fac737351230 | https://github.com/user-attachments/assets/da6ef150-e0ce-4f4d-bbd5-afb0858c588f |

Both videos retain all 361 frames at their original pixel scale after cropping
the bottom label. Source MP4s: [input](assets/demo/input.mp4) ·
[output](assets/demo/output.mp4).

## Quick Start

**Requirements:** Linux x86-64, Python 3.11–3.13, an NVIDIA GPU, and a CUDA
13.0-compatible driver. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
before running these commands.

1. Clone the repository and install the locked dependencies:

   ```bash
   git clone https://github.com/kopperx/ReCaVSR.git
   cd ReCaVSR
   uv sync --locked
   ```

2. Download the default model weights into `checkpoints/`:

   ```bash
   uvx hf download kopper/ReCaVSR \
     transformer.safetensors prompt.safetensors flashdecoder.safetensors \
     --local-dir checkpoints
   ```

3. Run a 31-frame preview:

   ```bash
   uv run python inference.py \
     --input assets/demo/input.mp4 \
     --output outputs/demo_preview_x4.mp4 \
     --scale 4 \
     --frames 31
   ```

The first run may take longer because compilation is enabled by default. To
process the full demo, omit `--frames` and choose a new `--output` path; the
script does not overwrite existing results.

## Pretrained Models

The [Hugging Face repository](https://huggingface.co/kopper/ReCaVSR) hosts the
weights. The matching JSON configuration files are already in `checkpoints/`.

| Weight file | Role | Needed for |
| --- | --- | --- |
| `transformer.safetensors` | Merged diffusion transformer | All runs |
| `prompt.safetensors` | Prompt embedding | All runs |
| `flashdecoder.safetensors` | Low-resolution-conditioned decoder | Default decoder |
| `vae.safetensors` | Original Wan decoder | Optional `--decoder wan` |

The transformer checkpoint is already merged; no adapter conversion is needed.
If you use a different `--model-dir`, copy `model_config.json`, `vae_config.json`,
and `flashdecoder_config.json` into it as well.

For the optional Wan decoder, download its weight and pass `--decoder wan` to
the inference command:

```bash
uvx hf download kopper/ReCaVSR vae.safetensors --local-dir checkpoints
```

## Inference Options

- `--scale 2` or `--scale 1.5` changes the spatial upscaling factor.
- `--device cuda:1` selects a different GPU.
- `--color-fix wavelet` or `--color-fix adain` enables color correction.
- `--model-dir /path/to/checkpoints` selects another model directory.
- `--frames 31` limits the run to a short preview.
- `--no-compile-blocks --no-compile-decoder` disables compilation.
- `--fp8-dit` enables faster, lossy FP8 calculations in DiT linears on NVIDIA
  GPUs with compute capability 8.9 or later. It requires compiled blocks;
  attention, KV caches and the decoder keep their original precision. Strip
  intermediates use `.fp8.stripN.mp4` names so BF16 strips are never reused by
  an FP8 run. The original checkpoints are unchanged.
- `--nvfp4-dit` enables faster, lossy NVFP4 DiT linears on Blackwell GPUs
  (tested on RTX 5070). It requires a PyTorch build with `_scaled_mm_v2`
  and compiled blocks. It is mutually exclusive with `--fp8-dit`; omitting
  both keeps original BF16 calculations. Attention, caches, conditioning and
  decoder retain their original precision. Intermediates use
  `.nvfp4.stripN.mp4` names, independently of FP8/BF16 strips. Weights are
  quantized in memory; the checkpoints are unchanged.

On a 300-frame DVD strip at 4x, NVFP4 reduced combined steady GPU time by
20.35% versus FP8 (1.256x throughput). Decoded output measured 30.76 dB PSNR
against FP8, so it is a quality/speed tradeoff, not equivalent output. This
is a strip benchmark, not a measured full-video wall-time improvement; first
compilation can outweigh the savings on a short clip.

Input may also be a numbered image directory; use `--fps` to set its playback
rate. See `uv run python inference.py --help` for the full CLI reference.

## Method at a Glance

1. **Recycled latents:** predictions from earlier blocks provide local temporal
   context to later blocks.
2. **Layer-wise cache routing:** transformer layers use different history scopes
   under a fixed cache budget.
3. **LR-conditioned FlashDecoder:** low-resolution observations help decode the
   generated latents efficiently.

For the architecture and experiments, see the [paper](https://arxiv.org/abs/2609.37831).

## Scope and Limitations

- The runtime requires a CUDA GPU and uses one GPU per inference process.
- Model inference is streaming, while the current reader and writer buffer the
  entire clip in CPU memory.
- Output preserves the input frame count but does not contain audio.
- The script writes an MP4 and a JSON run report, and refuses to overwrite
  either existing output.

## Citation

If ReCaVSR is useful in your research, please cite the paper:

```bibtex
@misc{wang2026recavsronestepstreamingdiffusion,
  title={ReCaVSR: One-Step Streaming Diffusion Video Super-Resolution with Recycled Latents and Learned Cache Routing},
  author={Xijun Wang and Xin Li and Suhang Yao and Zirui Lang and Bingchen Li and Zhibo Chen},
  year={2026},
  eprint={2609.37831},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2609.37831}
}
```

## Acknowledgements

This project builds on Wan and Hugging Face Diffusers. Color correction follows
[StableSR](https://github.com/IceClear/StableSR).

## License

The code is released under the [Apache License 2.0](LICENSE). Model weights and
demo footage remain subject to their respective licenses.
