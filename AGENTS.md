# AGENTS.md — mlputils

## Package overview

`mlputils` is a Python package for the full lifecycle of machine learning interatomic potentials (MLPs): generating DFT reference data, creating train/test datasets, training potentials (offline active learning), and benchmarking predictions against DFT. Target materials are FCC metals (Cu, Au, Al) and their nanoparticles.

Install: `pip install -e .` from this directory.

## Module map

```
reference_dft_maker.py  →  generates QE inputs + parses DFT outputs
        ↓ (reference data)
    makesets.py         →  splits configs into train/test/validation sets
        ↓ (datasets)
offline_training.py     →  trains FLARE/Sparse GP potentials (offline active learning)
        ↓ (trained model)
    benchmark.py        →  validates MLP against DFT properties
phonons_fd_test.py      →  phonon dispersion via finite differences (partially obsolete)
```

Each module can also be run standalone from the command line (see Entry points below).

## Module details

### `reference_dft_maker.py`

Generates Quantum ESPRESSO input files for DFT reference calculations and parses outputs.

**Input generation functions** (write `.pwi` files):
- `input_iso` — isolated atom (spin-polarized)
- `input_eos` — equation of state (volume scan)
- `input_fcc_relax` — vc-relax of bulk FCC
- `input_surfaces` — 111/110/100 surface slabs with layer fixing
- `input_isomers` — Ih/Dh/Oh clusters at 55 and 147 atoms
- `input_dimers` — dimer separation scan
- `input_phonons` — phonopy-generated displaced supercells

**Convergence study functions:**
- `convergence_ewfc_input_maker` — plane-wave cutoff scan
- `convergence_dual_input_maker` — dual (ecutrho/ecutwfc) scan
- `convergence_kpoints_smearing_input_maker` — k-points + smearing scan

**Parsing:**
- `parse_qe_results(symbol)` — reads `.pwo` files, fits EOS, computes surface energies, writes `{symbol}_reference_data.yaml`
- `parse_phonons(files, phonon_file_in)` — reads QE forces, produces force constants

**CLI:** `python -m mlputils.reference_dft_maker input|parse|plot <symbol> [args]`

Requires `parameters.yml` and `parameters_relax.yml` in the working directory for input generation.

### `makesets.py`

Dataset splitting with multiple strategies. Driven by a YAML config dict.

**Splitting strategies** (set `dataset_style` in config):
- `random` — shuffle all, split by ratio (default)
- `ordered` — shuffle per class (bulk/surf/cluster), keep class order
- `injected` — ordered + periodic injection of selected configs
- `intact` — use pre-existing train/test files as-is
- `cnap` — random split then sort by CNAP diversity metric
- `gcn` — random split then sort by GCN standard deviation

**Key functions:**
- `make_random_sets(files, ...)` — random splitting, supports per-file or global
- `make_sequential_sets(bulk, surf, cluster, ...)` — ordered splitting
- `make_sparsely_injected_sets(files, injection_files, ...)` — injection strategy
- `sort_by_gcn(conf_set, cutoff)` — sort by GCN spread (uses `snow.descriptors.coordination`)
- `sort_by_cnap_number(frames, cutoff)` — sort by CNAP diversity (uses `snow.descriptors.cna`)
- `make_sets(config)` — main entry: takes a config dict, writes `train.xyz` and `test.xyz`

**CLI:** `python -m mlputils.makesets <config.yaml>`

### `offline_training.py`

Offline active-learning loop for FLARE Sparse Gaussian Process potentials. This module is **FLARE-specific** — it imports FLARE at the top level and will fail if `mir-flare` is not installed.

**Main loop** (`train_offline`):
1. Initialize GP with random environments from first config
2. For each training config: compute uncertainties, add high-uncertainty environments to sparse set
3. Periodically optimize hyperparameters (negative log-likelihood or Huber loss)
4. Log learning curves (energy/force MAE on test set)
5. Write final model JSON + mapping files

**Key functions:**
- `train_offline(config, train_set, test_set)` — main training loop
- `optimize_hyps(gp_model, ...)` — hyperparameter optimization via `scipy.optimize.minimize`
- `ase2flare(struct, config, descriptors)` — convert ASE Atoms to FLARE Structure
- `initialize_gp(...)` — create empty SparseGP with kernels and descriptors
- `write_to_json(...)` — export trained model as JSON for LAMMPS use
- `get_dft_data(conf)` — extract energy/forces/stress from ASE Atoms (handles stress convention conversion)

**CLI:** `python -m mlputils.offline_training <config.yaml>`

Config must include `flare_calc` section (see `workflows_examples/offline_setup.yaml`).

### `benchmark.py`

Comprehensive MLP validation suite. Driven by a YAML config.

**Calculator factory:**
```python
calc = get_calc(config)  # config must have "calculator" and "model_file" keys
```
Supported `calculator` values:
- `"flare_lammps"` — FLARE via LAMMPS pair style (recommended for FLARE)
- `"mace"` — MACECalculator (requires model path)
- `"mace_mp"` — MACE-MP foundation model (no model file needed)
- `"flare"` — native FLARE calculator (not yet implemented)
- `"nequip"` — not yet implemented

**Benchmarking functions:**
- `eos_fcc_fit(symbol, calc, alat)` — bulk EOS → lattice constant, cohesive energy, bulk modulus
- `low_index_surfen(symbol, calc, alat)` — surface energies for 111/110/100
- `compute_test_errors(calc, test_set_files, E_iso)` — MAE/RMSE on energy, forces, stresses
- `adsorbate_curve` / `dimer_curve` — detect ghost potential holes at large separations
- `clusters_excess_energy(...)` — excess energy across Ih/Dh/Oh clusters up to max_size atoms
- `energy_levels_crossings(file, calc, symbol, alat, cohesive_energy)` — isomer energy ordering inversions vs reference (default: excess energy mode; calc cohesive energy computed from relaxed bulk FCC)
- `compute_phonons(calc, ...)` — phonon dispersion via phonopy (requires phonopy)
- `MD_performance(atoms, calc)` — MD steps/second benchmark
- `predict_configs(calc, ...)` — predict on special configs with % difference vs DFT

**Main entry** (`main(config_file)`): runs the full benchmark pipeline and writes a `{calculator}_benchmark.yaml` results file.

**CLI:** `python -m mlputils.benchmark <config.yaml>`

### `phonons_fd_test.py`

Phonon dispersion via finite displacement method. The author notes this module is **partially obsolete** — functionality is being split between `benchmark.py` (MLP phonons) and `reference_dft_maker.py` (DFT phonons).

Three modes:
- `write_input` — write QE inputs for phonopy-generated displaced structures
- `read_output` — read QE outputs and compute phonon band structure
- `use_calc` — compute forces with any ASE calculator (MLP or DFT)

Uses `get_calc` from `benchmark.py`.

## External dependencies

| Package | Where used | Required? |
|---------|-----------|-----------|
| `ase` | Everywhere (atoms, IO, calculators, optimizers, EOS) | Yes |
| `numpy` | Everywhere | Yes |
| `scipy` | `offline_training.py` (optimize), `makesets.py` | Yes |
| `pyyaml` | Config file loading | Yes |
| `mir-flare` | `offline_training.py` (FLARE imports at top level) | For training only |
| `mace` | `benchmark.py` (imported inside `get_calc`) | For MACE models |
| `phonopy` | `benchmark.py`, `reference_dft_maker.py`, `phonons_fd_test.py` (imported inside functions) | For phonons |
| `snow` | `makesets.py` (GCN/CNAP sorting), `benchmark.py` (KL divergence in energy_levels_crossings) | For sorting features |
| `tqdm` | `benchmark.py`, `offline_training.py` | Optional (graceful fallback) |
| `lammps` Python | `benchmark.py` (`get_calc` for flare_lammps) | For LAMMPS-based potentials |

## YAML config conventions

All modules load config from a YAML file passed as a CLI argument. Key config keys:

**`benchmark.py` config** (see `workflows_examples/benchmark_setup.yaml`):
- `symbol`, `E_iso`, `fcc_lattice_constant`, `cohesive_energy`, `Bulk_modulus` — DFT references
- `111_surface_energy`, `110_surface_energy`, `100_surface_energy` — DFT surface refs
- `calculator`, `model_file` — potential to test
- `test_set_file`, `err_method` — test set and error metric

**`offline_training.py` config** (see `workflows_examples/offline_setup.yaml`):
- `flare_calc` — full FLARE GP configuration block
- `optimizer_options` — hyperparameter optimization settings
- `dataset_style`, `bulk_files`, `surf_files`, `clusters_files` — dataset definition
- `call_threshold`, `add_threshold` — active learning uncertainty thresholds
- `optimize_every`, `min_optimize`, `max_optimize` — optimization schedule

**`makesets.py` config**: shares dataset keys with `offline_training.py` config above.

## Conventions

- Single-specie systems only (multi-specie not implemented)
- Units: energies in eV, distances in Angstrom, stresses in ASE Voigt convention
- FLARE stress convention differs from ASE — `get_dft_data()` handles conversion
- Isolated atom energy (`E_iso`) is subtracted from DFT energies for per-atom comparisons
- Calculator-dependent isolated atom energy conventions are handled by computing the model's own `E_iso` internally
- Output files are written to the current working directory (no configurable output path)
- `phonons_fd_test.py` has hardcoded Cu defaults — not reusable without modification
