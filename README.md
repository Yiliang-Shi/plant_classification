#Plant Classification

This is a sample project for large scale training of neural networks.
The goal is to set this up as a base template that could scale to a large
number of experiments, with easy to understand organization.


Organization explanation:

```python
experiments/  # Contains experiment specific files
    configs/ # Contains config.yaml files storing all experiment configuration
lib/          # The main tf2 code involving ML pipeline
    models/                # Model Architectures
data/   # Stores downloaded data for training
results/ # Stores desired deliverable
```
### Use instruction

In an virtual environment, call `pip install -e .` to install editable version
of library.

Experiments can be ran from notebooks in /experiments
