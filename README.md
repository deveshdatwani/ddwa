
# Modular Deep Learning Training System

This project allows you to train any PyTorch model by specifying the model class path and configuration.

## How to Use

1. **Implement your model**
	- Create your model class in any module, e.g. `dl/models/my_model.py`.
	- Your model should inherit from `torch.nn.Module`.
	- See `dl/models/template_model.py` for a template.

2. **Train your model**
	- Run the training script with the full class path:
	  ```bash
	  python dl/train/train.py --model dl.models.template_model.TemplateModel --dataset cifar10 --config dl/train/config.yaml
	  ```
	- Replace `dl.models.template_model.TemplateModel` with your own model class path.

3. **Configuration**
	- Model and training parameters are set in the YAML config file (see `dl/train/config.yaml`).

## Example Model Template

See `dl/models/template_model.py`:

```python
import torch.nn as nn

class TemplateModel(nn.Module):
	 def __init__(self, input_dim=3*32*32, num_classes=10, **kwargs):
		  super().__init__()
		  self.fc = nn.Linear(input_dim, num_classes)
	 def forward(self, x):
		  x = x.view(x.size(0), -1)
		  return self.fc(x)
```


## Using Hugging Face or External Models

You can also use models from Hugging Face Transformers or other external libraries, as long as the model class is importable and its constructor matches your config.

### Example: Hugging Face ViT

1. **Install the transformers library**
	```bash
	pip install transformers
	```

2. **Run the training script with a Hugging Face model**
	```bash
	python dl/train/train.py --model transformers.AutoModelForImageClassification --config dl/train/config.yaml
	```

3. **Example config.yaml**
	```yaml
	model:
	  pretrained_model_name_or_path: "google/vit-base-patch16-224"
	# ... other config options ...
	```

**Note:**
- The model class path (e.g., `transformers.AutoModelForImageClassification`) must be importable.
- The config must provide the correct arguments for the model's constructor.

---

## Notes
- You can implement any model architecture and pass it to the trainer.
- The trainer is agnostic to the model implementation as long as it is a `torch.nn.Module`.
