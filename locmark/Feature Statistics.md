=== DINOv3-ConvNeXt Feature Statistics (layer=1, dim=192) ===
Samples: 102,400 spatial vectors from 100 images

[DC Bias]
  Mean vector L2 norm    : 0.7608
  Expected if zero-mean  : 0.0018  (CLT baseline: sigma*sqrt(d/N))
  Actual / Expected ratio: 414.5×  (>>1 = significant DC bias)
  Bias on random anchor  : 0.0549  (accidental cos_sim from DC alone)

[Channel Dominance]
  Channel std range      : 0.0119 ~ 0.1929  (ratio: 16.3×)
  Channel std mean       : 0.0424
  Channel std CV         : 0.3790  (0 = uniform)

Interpretation:
  - DC ratio 415× >> 1: features are strongly biased toward a mean direction
    → zero-mean centering removes this bias from the anchor
  - Std ratio 16×: dominant channels have 16× more variance than weak ones
    → Rademacher equalization prevents dominant channels from causing false alignment