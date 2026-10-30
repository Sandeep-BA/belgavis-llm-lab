# Which tier are you?

By Sandeep Belgavi.

| You have | Tier | Notes |
|---|---|---|
| NVIDIA GPU, 8 GB+ VRAM | GPU | Linux natively, Windows through WSL2 (vLLM does not run on native Windows). |
| NVIDIA GPU, under 8 GB | GPU-lite | Inference with 4-bit GGUF; training only for models up to ~1B parameters. |
| Apple Silicon Mac | CPU+ | PyTorch MPS for small training runs; llama.cpp / Ollama use Metal for inference. vLLM is not the path here. |
| Any other laptop | CPU | Tiny models and GGUF inference. Every lesson has this path. |
| Only a browser | Free cloud GPU | Use the lesson notebook on Colab or Kaggle. |

A lesson's doc states the measured time and memory for each tier it was tested on.
