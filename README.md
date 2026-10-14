# mlputils
A small python package with utilities to train and test machine learning potentials (at the moment, with flare and mace)
and to generate quantum-espresso inputs for dft reference values.
## Installation
you can install mlputils with `pip` after cloning this repo.

## Output folders

Offline training and benchmarking accept an optional top-level YAML entry:

```yaml
output_folder: ./results
```

The default is `./`. Relative paths are resolved from the working directory where
commands are run; absolute paths and `~` are supported. Missing directories are
created automatically. Input dataset, model, phonon, and pseudopotential paths
keep their existing interpretation.

Offline training places generated train/test sets, logs, learning curves, parity
data, JSON models, added structures, and FLARE/LAMMPS mapping files in this folder.
Benchmarking places its YAML report, EOS data, optimizer logs, parity/error files,
phonon exports, and cluster excess-energy data there.

For DFT reference input generation, add the same entry to `parameters.yml` or the
parameter dictionary passed to `input_*` functions. The CLI now reads one explicitly specified YAML file for all generated references. Generated `.pwi` files and phonon setup YAML go to the
configured folder. The convergence input makers accept an `output_folder` keyword
argument, defaulting to `./`. Existing DFT parsing output conventions are unchanged.

## DFT reference input generation

Run, for example, from `workflows_examples`:

```bash
python -m mlputils.reference_dft_maker input Cu parameters_relax.yaml
```

The parameter file supplies `vacuum`, `pseudo_dir`, and `output_folder`.
Use `kpts: gamma` for Gamma-only calculations, or a positive integer for the
conventional-cell Monkhorst-Pack mesh (rescaled for slabs and phonon supercells).
Use `koffset: null` or `0` for an unshifted mesh, `1` for shifts in all directions,
or a three-element list such as `[1, 0, 1]`. Gamma inputs ignore mesh offsets.
