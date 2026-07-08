# Ocean Reconstruction Experiment Log

---

# Experiment 01

Date:

Name:
Baseline

Purpose:
First implementation using reasonable hyperparameters because the paper does not specify them.

Hyperparameters

```text
Sequence Length = 3

Patch Size = 4

Embedding Dimension = 128

Heads = 8

Encoder Layers = 6

ConvLSTM Hidden = 64

Batch Size = 8

Learning Rate = 1e-4

Epochs = 100
```

Results

```text
Training Loss :

Validation Loss :

Temperature RMSE :

Salinity RMSE :
```

Notes

```text
Initial baseline.
```
## Model Development

### ConvLSTM Positional Encoder

Status: ✅ Complete

Tests:
- [x] Forward pass
- [x] Output shape verified
- [x] Gradient check