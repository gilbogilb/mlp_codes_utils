#get DFT parameters from a yaml file
#write inputs
#run pw.x from bash

import yaml

from ase.io import write
from ase.build import bulk, fcc111, fcc110, fcc100
from ase.cluster import Octahedron, Decahedron, Icosahedron
from ase import Atoms
from ase.units import Rydberg as ry

import numpy as np

import sys

def convergence_ewfc_input_maker(symbol, 
                                 alat, 
                                 pseudo_dir,
                                 pseudos,
                                 ewfc_range=(40, 45, 50, 55, 60, 65, 70, 80, 90, 100, 120), #in Ry!!!!
                                 dual=8, 
                                 smearing=0.008, #in Ry!!!
                                 kpts=(8,8,8),
                                 conv_thr=1e-6/ry):
    
    #make inputs for convergence calculations: ecutwfc

    bulkfcc = bulk(symbol, 'fcc', alat, cubic=True)

    input_data = {
        'control': {
            'calculation': 'scf',
            'pseudo_dir': pseudo_dir,
        },
        'system': {
            #'ecutwfc': ,
            #'ecutrho': ,
            'occupations': 'smearing',
            'smearing': 'cold',
            'degauss': smearing,
        },
        'electrons': {
            'mixing_beta': 0.4,
            'electron_maxstep': 500,
            'mixing_mode': 'TF',
            'conv_thr': conv_thr
        },
    }

    for ecut in ewfc_range:
        input_data['system']['ecutwfc'] = ecut
        input_data['system']['ecutrho'] = dual*ecut
        write(f'{symbol}_bulk_ecut_{ecut}.pwi', bulkfcc, input_data=input_data, kpts=kpts, pseudopotentials=pseudos)

    return

def convergence_dual_input_maker(symbol, 
                                 alat, 
                                 pseudo_dir,
                                 pseudos,
                                 ewfc, #in Ry!!!!
                                 dual_range=(6,8,10,12), 
                                 smearing=0.008, #in Ry!!!
                                 kpts=(8,8,8),
                                 conv_thr=1e-6/ry):
    
    #make inputs for convergence calculations: ecutwfc

    bulkfcc = bulk(symbol, 'fcc', alat, cubic=True)

    input_data = {
        'control': {
            'calculation': 'scf',
            'pseudo_dir': pseudo_dir,
        },
        'system': {
            'ecutwfc': ewfc,
            #'ecutrho': parameters.get('dual')*parameters.get('ecutwfc'),
            'occupations': 'smearing',
            'smearing': 'cold',
            'degauss': smearing,
        },
        'electrons': {
            'mixing_beta': 0.4,
            'electron_maxstep': 500,
            'mixing_mode': 'TF',
            'conv_thr': conv_thr*len(bulkfcc)
        },
    }

    for d in dual_range:
        input_data['system']['ecutrho'] = d*ewfc
        write(f'{symbol}_bulk_dual_{d}.pwi', bulkfcc, input_data=input_data, kpts=kpts, pseudopotentials=pseudos)

    return

def convergence_kpoints_smearing_input_maker(symbol,
                                            alat,
                                            pseudo_dir,
                                            pseudos,
                                            ewfc,
                                            dual,
                                            smearing_range=(0.5/ry, 0.2/ry, 0.1/ry, 0.05/ry, 0.01/ry, 0.002/ry),#Ry
                                            kpts_range=(6,8,10,12,14,18,22),
                                            conv_thr=1e-6/ry
                                            ):

    #make inputs for convergence calculations: ecutwfc

    bulkfcc = bulk(symbol, 'fcc', alat, cubic=True)
    pos = bulkfcc.get_positions()
    bulkfcc.set_positions(
        pos + np.random.normal(scale=0.05*alat, size=pos.shape)
    )

    input_data = {
        'control': {
            'calculation': 'scf',
            'pseudo_dir': pseudo_dir,
        },
        'system': {
            'ecutwfc': ewfc,
            'ecutrho': dual*ewfc,
            'occupations': 'smearing',
            'smearing': 'cold',
        },
        'electrons': {
            'mixing_beta': 0.4,
            'electron_maxstep': 500,
            'mixing_mode': 'TF',
            'conv_thr': conv_thr*len(bulkfcc)
        },
    }

    for k in kpts_range:
        for sm in smearing_range:
            input_data['system']['degauss'] = sm
            write(f'{symbol}_bulk_kpts_{k}x{k}x{k}_smearing_{sm}.pwi', bulkfcc, input_data=input_data, kpts=[k, k, k], pseudopotentials=pseudos)

    return


def input_iso(symbol, parameters, vacuum=10.0):
    #writes input for a spin-polarized calculation of an isolated atom.
    #vacuum can be passed as an argument or in the parameters file.
    #given that convergence is sometimes hard to reach, delicate parameters
    #are chosen and ideally i would like to perform a first looser calculation
    #and a second more precise one starting from the initially computed wfc
    #so this function should write two input files

    #generate structure
    iso_atom = Atoms([symbol],[[0.,0.,0.]], pbc=False)
    iso_atom.center(vacuum = parameters.get('vacuum', vacuum) )

    kpts    = None
    pseudos = parameters.get('pseudos') 

    input_data = {
        'control': {
            'calculation': 'scf',
            'pseudo_dir': pseudo_dir,
        },
        'system': {
            'ecutwfc': parameters.get('ecutwfc'),
            'ecutrho': parameters.get('dual')*parameters.get('ecutwfc'),
            'occupations': 'smearing',
            'smearing': 'cold',
            'degauss': parameters.get('degauss_eV')/ry,
            'nspin': 2,
            'starting_magnetization': 1.0
        },
        'electrons': {
            'mixing_beta': 0.05,
            'electron_maxstep': 1500,
            'mixing_mode': 'local-TF',
            'conv_thr': parameters.get('econv_eV_peratom') #just one atom
            #'starting_pot': 'file'
            #'starting_wfc': 'file'
        },
    }

    write(f'{symbol}_iso.pwi', iso_atom, input_data=input_data, kpts=kpts, pseudopotentials=pseudos)
    
    return

def input_eos(symbol, alat_0, pseudo_dir, parameters, range_perc=(-3,3), npoints=10):

    k = parameters.get('kpts')
    pseudos = parameters.get('pseudos')

    input_data = {
        'control': {
            'calculation': 'scf',
            'pseudo_dir': pseudo_dir,
        },
        'system': {
            'ecutwfc': parameters.get('ecutwfc'),
            'ecutrho': parameters.get('ecutwfc')*parameters.get('dual'),
            'occupations': 'smearing',
            'smearing': 'cold',
            'degauss': parameters.get('degauss_eV')/ry,
        },
        'electrons': {
            'mixing_beta': 0.4,
            'electron_maxstep': 500,
            'mixing_mode': 'TF',
            'conv_thr': parameters.get('econv_eV_peratom')/ry*4. #cubic fcc cell has 4 atoms
        },
    }

    alats = np.linspace(alat_0*(1.+range_perc[0]/100.), alat_0*(1.+range_perc[1]/100.), npoints)

    for a in alats:
        bulkfcc = bulk(symbol, 'fcc', a, cubic=True)
        write(f'{symbol}_fcc_{a}.pwi', bulkfcc, input_data=input_data, kpts=[k,k,k], pseudopotentials=pseudos)

    return

def input_fcc_relax(symbol, alat_0, pseudo_dir, parameters_relax):

    k = parameters_relax.get('kpts')
    pseudos = parameters_relax.get('pseudos')

    input_data = {
        'control': {
            'calculation': 'vc-relax',
            'pseudo_dir': pseudo_dir,
        },
        'system': {
            'ecutwfc': parameters_relax.get('ecutwfc'),
            'ecutrho': parameters_relax.get('ecutwfc')*parameters.get('dual'),
            'occupations': 'smearing',
            'smearing': 'cold',
            'degauss': parameters_relax.get('degauss_eV')/ry,
        },
        'electrons': {
            'mixing_beta': 0.4,
            'electron_maxstep': 500,
            'mixing_mode': 'TF',
            'conv_thr': parameters_relax.get('econv_eV_peratom')/ry*4. #cubic fcc cell has 4 atoms
        },
    }

    bulkfcc = bulk(symbol, 'fcc', alat_0, cubic=True)
    write(f'{symbol}_fcc_relax.pwi', bulkfcc, input_data=input_data, kpts=[k,k,k], pseudopotentials=pseudos)

    return

def kpts_surf_calculator(cell, kpts_equivalent_conventional, a_ref, floor=True):

    Lx = cell[0][0]
    Ly = cell[1][1]

    kx = int(kpts_equivalent_conventional*a_ref/Lx)
    ky = int(kpts_equivalent_conventional*a_ref/Ly)

    if floor==False:
        kx+=1
        ky+=1

    return [kx, ky, 1]

def input_surfaces(symbol, pseudo_dir, vacuum, parameters_relax, size=(1,1,8) ):
    #writes quantum espresso inputs for relax calculations of surfaces 
    #fcc 111 110 100 with dft parameters from parameters_relax.
    #Structures are created with the initial lattice constant parameters_relax[latticeconstant][symbol]
    #and kpts are rescaled to have the equivalent of parameters_relax[kpts] points in the conventional
    #fcc cell per planar direction

    ref_lattice = parameters_relax.get('latticeconstant')[symbol]
    kpts_equiv  = parameters_relax.get('kpts')
    pseudos = parameters_relax.get('pseudos')

    s111 = fcc111(symbol, size, a=ref_lattice, vacuum=vacuum)
    s110 = fcc111(symbol, size, a=ref_lattice, vacuum=vacuum)
    s100 = fcc111(symbol, size, a=ref_lattice, vacuum=vacuum)

    k111 = kpts_surf_calculator(s111.get_cell(), kpts_equiv, ref_lattice)
    k110 = kpts_surf_calculator(s110.get_cell(), kpts_equiv, ref_lattice)
    k100 = kpts_surf_calculator(s100.get_cell(), kpts_equiv, ref_lattice)

    nat = len(s111)

    input_data = {
        'control': {
            'calculation': 'relax',
            'pseudo_dir': pseudo_dir,
        },
        'system': {
            'ecutwfc': parameters_relax.get('ecutwfc'),
            'ecutrho': parameters_relax.get('ecutwfc')*parameters.get('dual'),
            'occupations': 'smearing',
            'smearing': 'cold',
            'degauss': parameters_relax.get('degauss_eV')/ry,
        },
        'electrons': {
            'mixing_beta': 0.4,
            'electron_maxstep': 500,
            'mixing_mode': 'TF',
            'conv_thr': parameters_relax.get('econv_eV_peratom')/ry*nat
        },
    }

    write(f'{symbol}_surf_111_{size[0]}x{size[1]}x{size[2]}_relax.pwi', s111, kpts=k111, pseudopotentials=pseudos, input_data=input_data)
    write(f'{symbol}_surf_110_{size[0]}x{size[1]}x{size[2]}_relax.pwi', s110, kpts=k110, pseudopotentials=pseudos, input_data=input_data)
    write(f'{symbol}_surf_100_{size[0]}x{size[1]}x{size[2]}_relax.pwi', s100, kpts=k100, pseudopotentials=pseudos, input_data=input_data)    

    return

def input_isomers(symbol, pseudo_dir, vacuum, parameters_relax):
    """
    write inputs for relaxations of ih, dh, oh at 55 and 147 atoms.
    use parameters_relax to get reference lattice constants.
    """

    ref_lattice = parameters_relax.get('latticeconstant')[symbol]
    conv_atom   = parameters_relax.get('econv_eV_peratom')
    pseudos     = parameters_relax.get('pseudos')

    input_data = {
        'control': {
            'calculation': 'relax',
            'pseudo_dir': pseudo_dir,
        },
        'system': {
            'ecutwfc': parameters.get('ecutwfc'),
            'ecutrho': parameters.get('dual')*parameters.get('ecutwfc'),
            'occupations': 'smearing',
            'smearing': 'cold',
            'degauss': parameters.get('degauss_eV')/ry,
        },
        'electrons': {
            'mixing_beta': 0.4,
            'electron_maxstep': 700,
            'mixing_mode': 'local-TF',
        },
    }

    #55-atoms
    ico  = Icosahedron(symbol, 3, ref_lattice)
    octa = Octahedron(symbol, 5, 2, ref_lattice)
    deca = Decahedron(symbol, 3, 3, 0, ref_lattice)

    ico.center(vacuum=vacuum)
    octa.center(vacuum=vacuum)
    deca.center(vacuum=vacuum)

    input_data['electrons']['conv_thr'] = conv_atom*55.

    write(f'{symbol}_Ih_{len(ico)}_relax.pwi',  ico,  input_data=input_data, kpts=None, pseudopotentials=pseudos)
    write(f'{symbol}_Oh_{len(octa)}_relax.pwi', octa, input_data=input_data, kpts=None, pseudopotentials=pseudos)
    write(f'{symbol}_Dh_{len(deca)}_relax.pwi', deca, input_data=input_data, kpts=None, pseudopotentials=pseudos)


    #147-atoms
    ico  = Icosahedron(symbol, 4, ref_lattice)
    octa = Octahedron(symbol, 7, 3, ref_lattice)
    deca = Decahedron(symbol, 4,4,0, ref_lattice)

    ico.center(vacuum=vacuum)
    octa.center(vacuum=vacuum)
    deca.center(vacuum=vacuum)

    input_data['electrons']['conv_thr'] = conv_atom*147.

    write(f'{symbol}_Ih_{len(ico)}_relax.pwi',  ico,  input_data=input_data, kpts=None, pseudopotentials=pseudos)
    write(f'{symbol}_Oh_{len(octa)}_relax.pwi', octa, input_data=input_data, kpts=None, pseudopotentials=pseudos)
    write(f'{symbol}_Dh_{len(deca)}_relax.pwi', deca, input_data=input_data, kpts=None, pseudopotentials=pseudos)

    return

def input_dimers(symbol, separation_range, npoints, pseudo_dir, vacuum, parameters):

    kpts    = None
    pseudos = parameters.get('pseudos') 

    input_data = {
        'control': {
            'calculation': 'scf',
            'pseudo_dir': pseudo_dir,
        },
        'system': {
            'ecutwfc': parameters.get('ecutwfc'),
            'ecutrho': parameters.get('dual')*parameters.get('ecutwfc'),
            'occupations': 'smearing',
            'smearing': 'cold',
            'degauss': parameters.get('degauss_eV')/ry,
        },
        'electrons': {
            'mixing_beta': 0.3,
            'electron_maxstep': 1500,
            'mixing_mode': 'local-TF',
            'conv_thr': parameters.get('econv_eV_peratom')*2
        },
    }

    #generate structures
    for d in np.linspace(separation_range[0], separation_range[1], npoints):
        dimer = Atoms([symbol]*2, [[0.,0.,0.], [0.,0.,d]])
        dimer.center(vacuum = vacuum) 

        write(f'{symbol}_iso.pwi', dimer, input_data=input_data, kpts=kpts, pseudopotentials=pseudos)
    
    return

if __name__=='__main__':

    if len(sys.argv)<3:
        sys.exit(f'usage: {sys.argv[0]} chemical_symbol pseudo_dir')

    with open('parameters.yml', 'r') as f:
        parameters = yaml.safe_load(f)

    with open('parameters_relax.yml', 'r') as f:
        parameters_relax = yaml.safe_load(f)

    symbol     = sys.argv[1] #chemical symbol
    pseudo_dir = sys.argv[2] #pseudopotentials directory 

