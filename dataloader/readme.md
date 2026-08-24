# Dataloader

`DeepDTARawData` reads the original JSON/pickle layout, integer-encodes SMILES and sequences, and exposes the paper 5-fold splits.

`DTAPairDataset` / `make_dataloader` yield batches with `drug`, `protein`, `affinity`, `drug_idx`, `protein_idx`, and similarity rows when those files exist.
