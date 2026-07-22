# mlputils

Tools for the full lifecycle of machine learning interatomic potentials: generating DFT reference data, creating train/test datasets, training potentials via offline active learning, and benchmarking predictions against DFT. Targets FCC metals (Cu, Au, Al) and their nanoparticles.

## Installation

```bash
pip install -e .
```

Optional dependencies are split by functionality:

| Extra | Package | Needed for |
|-------|---------|------------|
| `flare` | `mir-flare` | FLARE training |
| `mace` | `mace` | MACE models |
| `phonopy` | `phonopy` | Phonon calculations |
| `snow` | `snow` | GCN/CNAP sorting, KL divergence |
| `all` | all above | Everything |

```bash
pip install -e ".[flare]"     # FLARE only
pip install -e ".[mace]"      # MACE only
pip install -e ".[all]"       # everything
```

Additional optional packages (not pip-installable via extras):
- `tqdm` — progress bars (graceful fallback if missing)
- `lammps` Python bindings — LAMMPS-based FLARE calculator

## Module map

```
reference_dft_maker.py  →  generates QE inputs + parses DFT outputs
        ↓ (reference data)
    makesets.py         →  splits configs into train/test sets
        ↓ (datasets)
offline_training.py     →  trains FLARE Sparse GP potentials
        ↓ (trained model)
    benchmark.py        →  validates MLP against DFT properties
```

All modules are runnable from the command line:

```bash
python -m mlputils.reference_dft_maker input|parse|plot <symbol> [args]
python -m mlputils.makesets <config.yaml>
python -m mlputils.offline_training <config.yaml>
python -m mlputils.benchmark <config.yaml>
```

## Quick start

### 1. Generate DFT reference data

Create `parameters.yml` and `parameters_relax.yml` in your working directory (see `reference_dft_maker.py` docstrings for keys), then:

```bash
python -m mlputils.reference_dft_maker input chemical_symbol vacuum_for_surfaces_and_clusters /path/to/pseudos
# run pw.x on the generated .pwi files, then:
python -m mlputils.reference_dft_maker parse chemical_symbol isolated_atom_energy
```

Unfortunately isolated atom energy is hard to parse for ase if nspin=2, so you ahve to give it to the parser to compute reference values.
The code produces `{symbol}_reference_data.yaml` with reference data such as lattice constant, cohesive energy, bulk modulus, and surface energies.

### 2. Create train/test datasets

Prepare a YAML config and run:

```bash
python -m mlputils.makesets dataset_config.yaml
```

Outputs `train.xyz` and `test.xyz`. Supported splitting strategies: `random`, `ordered`, `injected`, `intact`, `cnap`, `gcn`.

### 3. Train a FLARE potential

```bash
python -m mlputils.offline_training training_config.yaml
```

Runs the offline active-learning loop: initializes a Sparse GP, iteratively adds high-uncertainty environments, periodically optimizes hyperparameters, and writes the final model JSON.

### 4. Benchmark against DFT

```bash
python -m mlputils.benchmark benchmark_config.yaml
```

Computes bulk EOS, surface energies, test set errors (MAE/RMSE), adsorbate/dimer curves, and optionally phonons, cluster excess energies, and MD performance. Outputs `{calculator}_benchmark.yaml`.

## Supported calculators

| Calculator | Key | Notes |
|-----------|-----|-------|
| FLARE via LAMMPS | `flare_lammps` | Recommended for FLARE models |
| MACE | `mace` | |
| MACE-MP | `mace_mp` | Foundation model |
| FLARE native | `flare` | Not yet implemented |
| NequIP | `nequip` | Not yet implemented |

Set `calculator` and `model_file` in your benchmark YAML config.

## YAML config reference

### Benchmark config (`benchmark_setup.yaml`)

These are reference values which are compared to those computed with the mlp during benchmarking. Most of these keys are consistent with the dictionary produced by reference_data_maker.py

| Key | Description |
|-----|-------------|
| `symbol` | Chemical symbol (e.g. `'Cu'`) |
| `E_iso` | DFT isolated atom energy |
| `fcc_lattice_constant` | Reference lattice constant (Angstrom) |
| `cohesive_energy` | Reference cohesive energy per atom (eV) |
| `Bulk_modulus` | Reference bulk modulus (GPa) |
| `111/110/100_surface_energy` | Reference surface energies (J/m^2) |
| `calculator` | Calculator type (see table above) |
| `model_file` | Path to trained model |
| `test_set_file` | Test set XYZ file(s) |
| `err_method` | `'mae'` or `'rmse'` |

### Training config (`offline_setup.yaml`)

| Key | Description |
|-----|-------------|
| `flare_calc` | Full FLARE GP configuration block |
| `dataset_style` | Splitting strategy |
| `bulk_files`, `surf_files`, `clusters_files` | Dataset input files |
| `call_threshold` | Uncertainty threshold to trigger oracle call |
| `add_threshold` | Uncertainty threshold to add environments to sparse set |
| `optimize_every` | Frequency of hyperparameter optimization |
| `optimizer_options` | Loss function, method, bounds, and iteration limits |

See `workflows_examples/` for complete config files.

## Dependencies

| Package | Used in | Required? | Extra |
|---------|---------|-----------|-------|
| `ase` | Everywhere | Yes | — |
| `numpy` | Everywhere | Yes | — |
| `scipy` | `offline_training.py`, `makesets.py` | Yes | — |
| `pyyaml` | Config loading | Yes | — |
| `mir-flare` | `offline_training.py` | For FLARE training | `flare` |
| `mace` | `benchmark.py` | For MACE models | `mace` |
| `phonopy` | `benchmark.py`, `reference_dft_maker.py` | For phonons | `phonopy` |
| `snow` | `makesets.py`, `benchmark.py` | For GCN/CNAP sorting | `snow` |
| `lammps` Python | `benchmark.py` | For LAMMPS-based FLARE | — |

## Conventions

- Single-specie systems only (multi-specie not implemented)
- Units: eV, Angstrom, ASE Voigt stress convention
- Output files written to the current working directory
- `reference_dft_maker.py` requires `parameters.yml` and `parameters_relax.yml` in the working directory
