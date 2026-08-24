# Preprocessing

Character dictionaries and encoding helpers from the original `datahelper.py`.

- `label_smiles` / `label_sequence`: integer encoding used by the paper
- `one_hot_smiles` / `one_hot_sequence`: one-hot variant
- `log_transform_kd`: Davis pKd transform
- `prepare_new_data`: convert `ligands.tab` + `proteins.fasta` + `Y.tab`
