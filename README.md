 # Phase 1 Abstract: #

## Thomas O’Sullivan, Dr. David Hart ##
## May, 2025 ##


Vision Transformers have demonstrated strong performance in computer vision tasks. They
rely heavily on positional encoding and single key projections within their attention mechanism. In this work, we propose a multi-key projection approach, where each attention head is assigned a spatial mask that restricts attention to a specific directional context (left, right, up, down, or identity). This design explicitly encodes spatial priors into the model and aims to improve training efficiency and generalization by guiding attention across diverse spatial regions. Our baseline is a standard ViT-small-patch16-224 model trained from scratch on the Food-101 dataset. To ensure fair comparisons, both the baseline and our proposed architecture are trained under identical hyperparameter settings. We are currently addressing overfitting challenges that arise when training ViTs from scratch on small datasets. We are systematically experimenting with architectural simplifications (transformer block reduction), learning rate scheduling strategies, weight decay adjustments, and weight dropout. The end goal is to benchmark these traditional regularization strategies against our multi-key projection model under consistent training regimes to assess their relative impact on generalization and convergence.

