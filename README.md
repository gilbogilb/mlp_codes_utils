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
python -m mlputils.reference_dft_maker input chemical_symbol input_dft_parameters.yaml
# run pw.x on the generated .pwi files, then:
python -m mlputils.reference_dft_maker parse chemical_symbol isolated_atom_energy
```

The parameter file supplies `vacuum`, `pseudo_dir`, and `output_folder`.
Use `kpts: gamma` for Gamma-only calculations, or a positive integer for the
conventional-cell Monkhorst-Pack mesh (rescaled for slabs and phonon supercells).
Use `koffset: null` or `0` for an unshifted mesh, `1` for shifts in all directions,
or a three-element list such as `[1, 0, 1]`. Gamma inputs ignore mesh offsets.

## Offline optimization rollback

```yaml
when_rollback: accept
```

`accept` is the default. If optimized hyperparameter magnitudes are nonfinite or
fall outside the existing strict `minhyps < abs(hyps) < maxhyps` limits, the learner
restores the parameters in use immediately before that optimization. Newly added
training structures and sparse environments stay in the GP. Matrices are rebuilt
using the restored parameters, and logs record the rollback and actual parameters
in use. `discard` is not implemented and is rejected before training begins.
