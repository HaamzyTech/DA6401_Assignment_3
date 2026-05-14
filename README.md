# DA6401 - Assignment 3: Implementing the Transformer for Machine Translation

## [WandB REPORT Link](https://wandb.ai/id24s800-indian-institute-of-technology-madras/DA6401_Assignment_3/reports/Implementing-the-Transformer-for-Machine-Translation--VmlldzoxNjg3Njk2NQ?accessToken=c2muhfsqvp3lgos6rk81ve5rvyfl0c38gog119hhgxovnam5d4ghpl5cqa4d7sc5)

## [Git Repository Link](https://github.com/HaamzyTech/DA6401_Assignment_3)


## Overview

In this assignment, you will implement the landmark architecture from the paper "Attention Is All You Need" from scratch using PyTorch. The goal is to develop a Neural Machine Translation (NMT) system capable of translating text from German to English using the Multi30k dataset.

## Project Structure

```text
assignment3/
├── requirements.txt
├── README.md
├── model.py           # Core Transformer architecture (Encoders, Decoders, Multi-Head Attention)
├── utils.py           # Label Smoothing, Noam Scheduler, Masking Utilities
├── dataset.py         # Multi30k dataset loading and spacy tokenization
├── train.py           # Training loops and Greedy Decoding inference
```

## 3. Required Experiment Runs

### Project Setup

Run the following commands in the root directory of the project.

```bash
# 1. create a virtual environment

python -m venv .venv

# 2. Activate the virtual environment

.venv/Scripts/activate      # Windows

source .venv/bin/activate   # Linux

# 3 install dependencies

pip install -r requirements.txt

```

### Run A: Baseline

This run is reused as the comparison point for Noam, scaling, sinusoidal positions, and label smoothing.

```powershell
python train.py `
  --run-name baseline_noam_scaled_sinusoidal_ls01 `
  --wandb-project da6401-a3 `
  --use-noam `
  --scale-attention `
  --sinusoidal-positional-encoding `
  --label-smoothing 0.1 `
  --batch-size 64 `
  --num-epochs 10 `
  --d-model 512 `
  --num-layers 6 `
  --num-heads 8 `
  --d-ff 2048 `
  --dropout 0.1 `
  --warmup-steps 4000 `
  --max-decode-len 100 `
  --checkpoint-path checkpoints/baseline.pt
```

### Run B: Fixed Learning Rate

Use this for the Noam scheduler ablation.

```powershell
.\.venv\Scripts\python.exe train.py `
  --run-name fixed_lr_scaled_sinusoidal_ls01 `
  --wandb-project da6401-a3 `
  --no-noam `
  --fixed-lr 0.0001 `
  --scale-attention `
  --sinusoidal-positional-encoding `
  --label-smoothing 0.1 `
  --batch-size 64 `
  --num-epochs 10 `
  --d-model 512 `
  --num-layers 6 `
  --num-heads 8 `
  --d-ff 2048 `
  --dropout 0.1 `
  --max-decode-len 100 `
  --checkpoint-path checkpoints/fixed_lr.pt
```

### Run C: No Scaling Factor

Use this for the `1 / sqrt(d_k)` ablation. W&B logs `grad_norm/query` and `grad_norm/key` for the first 1000 training steps.

```powershell
.\.venv\Scripts\python.exe train.py `
  --run-name no_scale_noam_sinusoidal_ls01 `
  --wandb-project da6401-a3 `
  --use-noam `
  --no-scale-attention `
  --sinusoidal-positional-encoding `
  --label-smoothing 0.1 `
  --batch-size 64 `
  --num-epochs 10 `
  --d-model 512 `
  --num-layers 6 `
  --num-heads 8 `
  --d-ff 2048 `
  --dropout 0.1 `
  --warmup-steps 4000 `
  --max-decode-len 100 `
  --checkpoint-path checkpoints/no_scale.pt
```

### Run D: Learned Positional Embeddings

Use this for sinusoidal positional encoding vs learned positional embeddings.

```powershell
.\.venv\Scripts\python.exe train.py `
  --run-name learned_pos_noam_scaled_ls01 `
  --wandb-project da6401-a3 `
  --use-noam `
  --scale-attention `
  --learned-positional-encoding `
  --label-smoothing 0.1 `
  --batch-size 64 `
  --num-epochs 10 `
  --d-model 512 `
  --num-layers 6 `
  --num-heads 8 `
  --d-ff 2048 `
  --dropout 0.1 `
  --warmup-steps 4000 `
  --max-decode-len 100 `
  --checkpoint-path checkpoints/learned_pos.pt
```

### Run E: No Label Smoothing

Use this for label smoothing sensitivity.

```powershell
.\.venv\Scripts\python.exe train.py `
  --run-name no_label_smoothing_noam_scaled_sinusoidal `
  --wandb-project da6401-a3 `
  --use-noam `
  --scale-attention `
  --sinusoidal-positional-encoding `
  --label-smoothing 0.0 `
  --batch-size 64 `
  --num-epochs 10 `
  --d-model 512 `
  --num-layers 6 `
  --num-heads 8 `
  --d-ff 2048 `
  --dropout 0.1 `
  --warmup-steps 4000 `
  --max-decode-len 100 `
  --checkpoint-path checkpoints/no_label_smoothing.pt
```
