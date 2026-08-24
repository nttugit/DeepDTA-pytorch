# Models

PyTorch modules in `models/deepdta.py`.

The default model is `DeepDTA` (`combined_categorical`): two 3-layer 1D-CNN encoders with global max pooling, concatenated into an FC regression head. Filter counts grow as `n, 2n, 3n` with `n=32` in the paper.
