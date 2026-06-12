#testing: verifying potentials by computing the phonon dispersion curve with finite difference method with dft and a potential.
#mostly got the code from the flare tutorial on active learning with lammps (https://colab.research.google.com/drive/1Syfd7s-SGHjYDHdSLKyvgIdszLQpssh2?usp=sharing#scrollTo=N8mtsv3yaq7H)
#this code should be obsolete and divided between benchmark.py and reference_dft_maker.py


import numpy as np
import matplotlib.pyplot as plt

from ase import Atoms
from ase.spacegroup import crystal
from ase.calculators.lammpsrun import LAMMPS
from ase.calculators.espresso import Espresso
from ase.io import write, read

from phonopy import Phonopy, load
from phonopy.structure.atoms import PhonopyAtoms

import os
import sys

from mlputils.benchmark import get_calc

import yaml

def plot_phonons_from_file(file, outname='band.yaml'):

    phonon = load(file)
    phonon.auto_band_structure(plot=True, write_yaml=True, filename=outname)
    plt.show()

def kpts_equiv(new_length, k_in_conventional, lattice_constant, ceil=True):
    if ceil:
        return int(1+k_in_conventional*lattice_constant/new_length)
    else:
        return int(k_in_conventional*lattice_constant/new_length)

#system settings
symbol = 'Cu'
alat   = 3.557041

#mode: write_input (write pw.x input), read_output (read forces from .pwo files), use_calc (use ase calculator with parameters defined in calc_config_file)
mode = sys.argv[1]

#DFT settings:
input_data = {
    'control': {
        'calculation': 'scf',
        'pseudo_dir': '/home/users/gilberto.nardi/pseudopotentials',
        'tprnfor': True,
        'tstress': False
    },
    'system': {
        'ecutwfc': 45,
        'ecutrho': 315,
        'occupations': 'smearing',
        'smearing': 'cold',
        'degauss': 0.008,
    },
    'electrons': {
        'mixing_beta': 0.4,
        'electron_maxstep': 500,
        'mixing_mode': 'TF',
        'conv_thr': 1e-9
    },
}
k_conv = 14 #k points per conventional cell

pseudos = {"Cu": "Cu.pbesol-dn-rrkjus_psl.1.0.0.UPF", 
           "Au": " Au.pbesol-dn-rrkjus_psl.0.3.0.UPF",
           "Al": "Al.pbesol-n-rrkjus_psl.1.0.0.UPF"}

#calc settings
calc_config_file = 'phonons_setup.yaml'


#make an ase structure
cell = crystal(symbol, [(0.,0.,0.)], spacegroup=225, cellpar=[alat, alat, alat, 90., 90., 90.], primitive_cell=True)

#phonopy stuff 
phcell = PhonopyAtoms(cell=cell.get_cell(), positions=cell.get_positions(), numbers=cell.get_atomic_numbers())
phonon = Phonopy(phcell, supercell_matrix=np.eye(3)*4)
phonon.generate_displacements(distance=0.02)

#for calc in (dftcalc, flarecalc):
forces = []
for i, supercell in enumerate(phonon.supercells_with_displacements):
    atoms = Atoms(cell=supercell.cell, numbers=supercell.numbers, positions=supercell.positions, pbc=True)
    
    if mode=='write_input':

        kx = kpts_equiv(atoms.cell.lengths()[0], k_conv, alat, ceil=False)
        ky = kpts_equiv(atoms.cell.lengths()[1], k_conv, alat, ceil=False)
        kz = kpts_equiv(atoms.cell.lengths()[2], k_conv, alat, ceil=False)

        print(kx, ky, kz)

        write(f'phonopy_structure_{i}.pwi', atoms, input_data=input_data, kpts=[kx,ky,kz], pseudopotentials=pseudos, koffset=(1,1,1) )

    elif mode == 'read_output':

        atoms = read(f'phonopy_structure_{i}.pwo')
        forces.append(atoms.get_forces())

    elif mode == 'use_calc':

        with open(calc_config_file) as f:
            calc_config = yaml.safe_load(f)
        calc = get_calc(calc_config)
        atoms.calc = calc

        forces.append(atoms.get_forces()) #computes forces along the way
    
    else:

        sys.exit(f'unrecognized mode {mode}')


if mode!='write_input':

    file_out = f"{symbol}_phonopy_params.yaml"

    phonon.forces = forces
    phonon.produce_force_constants()
    phonon.save(file_out, settings={"force_constants": True})

    plot_phonons_from_file(file_out, outname=f'band_{mode}.yaml')

