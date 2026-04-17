#get DFT parameters from a yaml file
#write inputs
#run pw.x from bash

import yaml

from ase.io import write, read
from ase.build import bulk, fcc111, fcc110, fcc100
from ase.cluster import Octahedron, Decahedron, Icosahedron
from ase import Atoms
from ase.units import Rydberg as ry
from ase.units import kJ
from ase.eos import EquationOfState

import numpy as np

import sys
import os
import glob

#convergence study makers
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
            'tstress' : True,
            'tprnfor': True,
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

#dft benchmarks makers
def input_iso(symbol, pseudo_dir, parameters, vacuum=10.0, smearing_divider=1.0):
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
            'degauss': parameters.get('degauss_eV')/ry/smearing_divider,
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
        write(f'{symbol}_fcc_{a:.3f}.pwi', bulkfcc, input_data=input_data, kpts=[k,k,k], pseudopotentials=pseudos)

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
            'ecutrho': parameters_relax.get('ecutwfc')*parameters_relax.get('dual'),
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
    s110 = fcc110(symbol, size, a=ref_lattice, vacuum=vacuum)
    s100 = fcc100(symbol, size, a=ref_lattice, vacuum=vacuum)

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
            'ecutrho': parameters_relax.get('ecutwfc')*parameters_relax.get('dual'),
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
            'ecutwfc': parameters_relax.get('ecutwfc'),
            'ecutrho': parameters_relax.get('dual')*parameters_relax.get('ecutwfc'),
            'occupations': 'smearing',
            'smearing': 'cold',
            'degauss': parameters_relax.get('degauss_eV')/ry,
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

def input_dimers(symbol, separation_range, npoints, pseudo_dir, vacuum, parameters, smearing_divider=1.0):

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
            'degauss': parameters.get('degauss_eV')/ry/smearing_divider,
        },
        'electrons': {
            'mixing_beta': 0.05,
            'electron_maxstep': 1500,
            'mixing_mode': 'local-TF',
            'conv_thr': parameters.get('econv_eV_peratom')*2
        },
    }

    #generate structures
    for d in np.linspace(separation_range[0], separation_range[1], npoints):
        dimer = Atoms([symbol]*2, [[0.,0.,0.], [0.,0.,d]])
        dimer.center(vacuum = vacuum) 

        write(f'{symbol}_dimer_{d:.3f}.pwi', dimer, input_data=input_data, kpts=kpts, pseudopotentials=pseudos)
    
    return

def parse_qe_results(symbol, E_iso_ry=None, directory=".", surf_size='1x1x8'):
    """
    Parse Quantum ESPRESSO output files and compute:
    - E_iso: energy of isolated atom
    - fcc_lattice_constant: equilibrium lattice constant from EOS fit
    - cohesive_energy: cohesive energy per atom
    - Bulk_modulus: from EOS fit
    - 111/110/100 surface energies
    """
    results = {}

    # --- Isolated atom ---
    #having some problems due to the magnetic moment not being read quite correctly.
    #will be able to insert the isolated atom energy (in rydberg) from commandline
    try:
        iso = read(os.path.join(directory, f"{symbol}_iso.pwo"))
        E_iso = iso.get_potential_energy()
        results["E_iso"] = E_iso
    except Exception as e:
        print(f'Could not read the isolated atom file - {e}. Trying with user-specified argument...')
        if E_iso_ry is not None:
            E_iso = E_iso_ry*ry
            results["E_iso"] = E_iso
        else:
            sys.exit('Could not read isolated atom energy from the .pwo file. Please provide it as argument to the qe parser function (generally ref_maker.py symbol E_iso_ry (in Rydberg!!))')

    # --- EOS fit for bulk properties ---
    # Collect all single-point fcc calculations (fixed lattice constants)
    eos_files = sorted(glob.glob(os.path.join(directory, f"{symbol}_fcc_*.pwo")))
    eos_files = [f for f in eos_files if "relax" not in f]

    volumes, energies = [], []
    for f in eos_files:
        try:
            atoms = read(os.path.join(directory, f))
            volumes.append(atoms.get_volume())
            energies.append(atoms.get_potential_energy())
        except Exception as e:
            print(f'Could not read {fname}: {e}')

    #fit
    eos = EquationOfState(volumes, energies, eos="murnaghan")
    v0, e0, B = eos.fit()

    #do some conversions
    a0 = (4 * v0/len(atoms)) ** (1 / 3)
    results["fcc_lattice_constant"] = float(a0)
    results["a0_eos"] = float(a0)
    # Bulk modulus: ASE returns it in eV/Å^3, convert to GPa
    results["Bulk_modulus"] = float(B / kJ * 1e24)

    #write e0 from eos
    results["ecoh_eos"] = float(e0/len(atoms)-E_iso)

    #write eos to file
    alats = [(4*v/len(atoms)) ** (1/3) for v in volumes]
    np.savetxt(f'{symbol}_eos.dat', np.column_stack((alats, energies)))
    results['eos_file'] = f'{symbol}_eos.dat'

    # Use the relaxed bulk as reference
    bulk_relax = read(os.path.join(directory, f"{symbol}_fcc_relax.pwo"))
    N_bulk = len(bulk_relax)
    E_bulk = bulk_relax.get_potential_energy()
    E_coh = (E_bulk / N_bulk) - E_iso
    results["cohesive_energy"] = float(E_coh)
    results["ecoh_relax"] = float(E_coh)
    results["a0_relax"] = float( bulk_relax.get_cell().volume ** (1./3.)   )

    # --- Surface energies ---
    # Surface energy = (E_slab - N_slab * E_bulk_per_atom) / (2 * A)
    # Factor 2 because slab has two surfaces
    E_bulk_per_atom = E_bulk / N_bulk

    surface_tags = {
        "111": f"{symbol}_surf_111_{surf_size}_relax.pwo",
        "110": f"{symbol}_surf_110_{surf_size}_relax.pwo",
        "100": f"{symbol}_surf_100_{surf_size}_relax.pwo",
    }

    for miller, fname in surface_tags.items():
        try:
            slab = read(os.path.join(directory, fname))
            N_slab = len(slab)
            E_slab = slab.get_potential_energy()
            cell = slab.get_cell()
            # Surface area from cross product of two in-plane lattice vectors
            A = np.linalg.norm(np.cross(cell[0], cell[1]))
            E_surf = (E_slab - N_slab * E_bulk_per_atom) / (2 * A)
            # Convert eV/Å^2 to J/m^2
            E_surf_Jm2 = E_surf * 16.0218
            results[f"{miller}_surface_energy"] = float(E_surf_Jm2)
        except Exception as e:
            print(f'Could not read {fname}: {e}')

    # --- dimers ---
    dimers_files = sorted(glob.glob(os.path.join(directory, f"{symbol}_dimer_*.pwo")))
    dimer_sep, dimer_ene = [], []
    for f in dimers_files:
        #check if the calculation worked - it might have not converged...
        try:
            atoms = read(f)
            d = np.linalg.norm(atoms.positions[1]-atoms.positions[0])
            dimer_sep.append(d)
            dimer_ene.append(atoms.get_potential_energy())
        except Exception as e:
            print(f'Could not read data from {f} ({e}). It is possible the calculation did not converge...')
    
    np.savetxt(f'{symbol}_dimer_dft.dat', np.column_stack( (dimer_sep, dimer_ene) ) )
    results["dimer_curve_file"] = f'{symbol}_dimer_dft.dat'

    # --- Write to YAML ---
    out_path = os.path.join(directory, f"{symbol}_reference_data.yaml")
    with open(out_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)

    print(f"Results written to {out_path}")
    return results


#main wrappers
def write_all_inputs(args): #args=sys.argv

    if len(args)<4:
        sys.exit(f'usage: {args} chemical_symbol vacuum pseudo_dir')

    with open('parameters.yml', 'r') as f:
        parameters = yaml.safe_load(f)

    with open('parameters_relax.yml', 'r') as f:
        parameters_relax = yaml.safe_load(f)

    symbol     = args[1] #chemical symbol
    vacuum     = float(args[2]) #vacuum for surfs
    pseudo_dir = args[3] #pseudopotentials directory 
    

    alat_0     = parameters.get('latticeconstant')[symbol]

    input_iso(symbol, pseudo_dir, parameters, smearing_divider=100.)
    input_eos(symbol, alat_0, pseudo_dir, parameters)
    input_fcc_relax(symbol, alat_0, pseudo_dir, parameters_relax)
    input_surfaces(symbol, pseudo_dir, vacuum, parameters_relax)
    input_isomers(symbol, pseudo_dir, vacuum, parameters_relax)
    input_dimers(symbol, separation_range=(alat_0/1.412*0.65, alat_0/1.412*3.0), npoints=10, pseudo_dir=pseudo_dir, vacuum=vacuum, parameters=parameters, smearing_divider=100.)

if __name__=='__main__':

    if not (sys.argv[1]!='input' or sys.argv[1]!='parse'):
        print(f'USAGE: {sys.argv[0]} <mode> <args>')
        print('--------------------------')
        print('if <mode>==input, args should be: <chemical_symbol> <vacuum_for_surfaces_and_clusters> <pseudopotenials_directory>')
        print('and parameters.yml and parameters_relax.yml should be present in the execution folder.')
        print('--------------------------')
        print('if <mode>==parse, args should be: <chemical_symbol>')

    if sys.argv[1] == 'input':
        write_all_inputs(sys.argv)
    
    elif sys.argv[1] == 'parse':
        if len(sys.argv)>2:
            parse_qe_results(symbol=sys.argv[1], E_iso_ry=float(sys.argv[2]) )
        else:
            parse_qe_results(sys.argv[1])
