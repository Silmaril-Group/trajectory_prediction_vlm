---
title: Duck Hunt VLM Demo
emoji: "\U0001F3AF"
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "5.29.0"
app_file: app.py
python_version: "3.12"
models:
  - dmayboroda/dh_ministal_gpro
  - mistralai/Ministral-3-8B-Instruct-2512-BF16
pinned: false
---

# Duck Hunt VLM Demo

Watch an AI agent play Duck Hunt! The agent predicts duck trajectories accounting for processing latency, and fires.

- **Trained model**: [dmayboroda/dh_ministal_gpro](https://huggingface.co/dmayboroda/dh_ministal_gpro) (Ministral-3B + LoRA, 60.9% hit rate via GRPO)
- **Input**: Sequential game frames with duck positions
- **Output**: Normalized (x, y) shot coordinates + horizon prediction
