# CDFDoubleGridNet

## Overview

**CDFDoubleGridNet** is a multi-scale voxel U-Net with temporal attention and CDF grid warping in each axis.

---

## Architecture

| Component | Detail |
|-----------|--------|
| **Temporal encoder** | Per-point linear projection (3→32) → 4-head self-attention across T=5 frames, applied independently at each spatial point |
| **Fourier positional encoding** | 8-band sinusoidal encoding of position-normalized coordinates; frequencies 2^k·π, k=0…7 |
| **SDF embedding** | 2-layer MLP mapping `(sdf/5, log1p(sdf·10)/2.4)` → 16-D geometry-aware feature |
| **Multi-scale voxel U-Net** | Two-level CDF-warped voxel pyramid (32³ coarse + 80³ fine), each level: scatter-mean into voxels → 1×1 Conv3D context fusion → 3-level U-Net → trilinear sample-back |
| **CDF grid warping** | Gaussian-smoothed CDF mapping warps physical positions into uniform computational coordinates, concentrating voxel resolution near the boundary layer and wake |
| **Residual prediction** | Output = last input frame + predicted delta |
| **No-slip enforcement** | Airfoil surface points are zeroed in the output (hard physical constraint) |
| **TTA** | Y-axis flip test-time averaging (two forward passes averaged at inference) |

**Parameter count:** 33.78 M

---

## Normalization (handled inside model.py)

All normalization statistics are computed on the training set, stored as registered buffers (`vel_mean`, `vel_std`, `pos_mean`, `pos_std`), and saved inside `state_dict.pt`. They are automatically applied at inference — **no caller pre-processing is needed**.

The model accepts raw, unscaled data directly in the competition signature:

```python
from models.cdf_2grid.model import Model
model = Model()   # downloads from hf, loads weights and normalization from state_dict.pt automatically
velocity_out = model(t, pos, idcs_airfoil, velocity_in)
```

### Training objective

Relative L2 loss + 0.01 × absolute L2 loss. Autoregressive pushforward training on 50 % of batches.

---

## Dependencies

```
torch >= 2.1
numpy
pyyaml
huggingface_hub
```
