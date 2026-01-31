

# Modular Deep Learning Training System

This project provides a modular, flexible framework for training, validating, and deploying PyTorch models with:
- Custom, cloned, or Hugging Face models
- Configurable loss functions
- YAML and CLI configuration (CLI overrides YAML)
- Weights & Biases (wandb) experiment tracking
- User confirmation before each run

## Quick Start

1. **Add Your Model**
	 - Place custom models in `dl/models/custom/`, cloned models in `dl/models/cloned/`.
	 - Each model should be a `torch.nn.Module` class.
	 - See `dl/models/template_model.py` for a template.

2. **Configure Your Experiment**
	 - Edit `dl/train/config.yaml` to set:
		 - Model source/name and parameters
		 - Loss function type/params
		 - Dataset and run settings
		 - wandb project and plot name

3. **Run Training**
	 - Use the CLI to override any config option:
		 ```bash
		 python dl/train/train.py --model__source custom --model__name MyModel --loss__type focal --wandb__plot_name my_exp
		 ```
	 - The trainer will print all settings and ask for confirmation before starting.

4. **Weights & Biases Integration**
	 - Set your wandb API key (see wandb docs) or run `wandb login` once.
	 - All runs are logged to wandb with your chosen plot name.

## Configuration Example (`dl/train/config.yaml`)

```yaml
model:
	source: custom         # custom, cloned, huggingface
	name: vit              # model name or huggingface id
	depth: 10
	in_dim: 192
	inner_dim: 128
	num_classes: 10
	img_size: 32
	patch_size: 8
	in_channels: 3

loss:
	type: cross_entropy    # cross_entropy, focal, etc.
	params: {}

wandb:
	project: dl-training
	plot_name: experiment_1

dataset:
	name: cifar10
	path: ../data/cifar-10-batches-py

run:
	batch_size: 32
	lr: 0.0001
	epochs: 10
	save_path: "./checkpoint/cifar_transformer.pth"
	device: "cpu"
	quantize: false
	quantized_save_path: "./checkpoint/model_quantized.pth"
```

## Features

- **Modular Models:** Add custom or cloned models easily. Hugging Face models supported.
- **Flexible Loss:** Choose loss function (cross-entropy, focal, etc.) via config/CLI.
- **Configurable Everything:** All options in YAML and CLI (CLI takes precedence).
- **User Confirmation:** Trainer prints config and asks for y/n before running.
- **wandb Logging:** Set project/plot name in config or CLI for clear experiment tracking.
- **Quantization:** Optional post-training quantization.

## Directory Structure

- `dl/models/custom/` — Your custom models
- `dl/models/cloned/` — Models cloned from external sources
- `dl/utils/` — Trainer, loss functions, dataloaders
- `dl/train/` — Training scripts and configs

## Example: Add a Custom Model

1. Create `dl/models/custom/MyModel.py`:
	 ```python
	 import torch.nn as nn
	 class MyModel(nn.Module):
			 def __init__(self, ...):
					 super().__init__()
					 # ...
			 def forward(self, x):
					 # ...
	 ```
2. Set `model.source: custom` and `model.name: MyModel` in config or CLI.

## Example: Use a Hugging Face Model

1. Install transformers:
	 ```bash
	 pip install transformers
	 ```
2. Set `model.source: huggingface` and `model.name: <hf-model-id>` in config or CLI.

## Notes
- All config options can be set in YAML or overridden via CLI (e.g., `--loss__type focal`).
- The trainer is fully modular and will prompt for confirmation before running.
- For wandb, set your API key once with `wandb login` or via `WANDB_API_KEY` env variable.
