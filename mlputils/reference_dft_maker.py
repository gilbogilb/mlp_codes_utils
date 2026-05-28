#get DFT parameters from a yaml file
#write inputs
#run pw.x from bash
# OR
# parse results from calcualtions

import yaml

from ase.io import write, read
from ase.build import bulk, fcc111, fcc110, fcc100
from ase.cluster import Octahedron, Decahedron, Icosahedron
from ase import Atoms
from ase.units import Rydberg as ry
from ase.units import kJ
from ase.eos import EquationOfState
from ase.spacegroup import crystal
from ase.constraints import FixAtoms

import numpy as np

import sys
import os
import glob
import re

################################
### convergence study makers ###
################################

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


#############################
### dft benchmarks makers ###
#############################

def input_from_yaml(parameters, nat, pseudo_dir, calc_type='scf', electron_maxstep=500, mixing_mode='plain'):
    """
    standard input_data maker from a yaml file. A few extra args are available, with default values.
    nat: number of atoms in the system (energy convergence thresholds are extensive)
    pseudo_dir: directory in which you have psuedopotential files   

    """

    input_data = {
        'control': {
            'calculation': calc_type,
            'pseudo_dir': pseudo_dir,
            'tstress': True,
            'tprnfor': True,
            'etot_conv_thr': parameters.get('etot_conv_thr_eV_peratom',1e-4)/ry*nat, #not used if not relaxing
            'forc_conv_thr': parameters.get('forc_conv_thr', 1e-3)                   #not used if not relaxing
        },
        'system': {
            'ecutwfc': parameters.get('ecutwfc'),
            'ecutrho': parameters.get('ecutwfc')*parameters.get('dual'),
            'occupations': 'smearing',
            'smearing': parameters.get('smearing', 'mv'),
            'degauss': parameters.get('degauss_eV')/ry,
        },
        'electrons': {
            'mixing_beta': parameters.get('mixing_beta', 0.7),
            'electron_maxstep': electron_maxstep,
            'mixing_mode': mixing_mode,
            'conv_thr': parameters.get('econv_eV_peratom')/ry*nat
        },
    }

    return input_data

def input_iso(symbol, pseudo_dir, parameters, nspin=2, starting_magnetization=1.0, vacuum=10.0, smearing_divider=1.0, mixing_beta=None):
    #writes input for a spin-polarized calculation of an isolated atom.
    #vacuum can be passed as an argument or in the parameters file.
    #given that convergence is sometimes hard to reach, delicate parameters
    #are chosen and ideally i would like to perform a first looser calculation
    #and a second more precise one starting from the initially computed wfc
    #so this function should write two input files
    #smearing divider eventually further reduces the smeraing passed from parameters
    #as large smearing makes convergence very hard

    #generate structure
    iso_atom = Atoms([symbol],[[0.,0.,0.]], pbc=False)
    iso_atom.center(vacuum = parameters.get('vacuum', vacuum) )

    kpts    = None
    pseudos = parameters.get('pseudos') 

    input_data = input_from_yaml(parameters, len(iso_atom), pseudo_dir, electron_maxstep=1500, mixing_mode='local-TF')

    #some case specific options for the input
    input_data['system']['nspin'] = nspin
    if nspin == 2:
        input_data['system']['starting_magnetization'] = starting_magnetization

    input_data['system']['degauss'] = input_data['system']['degauss']/smearing_divider

    if mixing_beta is not None:
        input_data['electrons']['mixing_beta'] = mixing_beta

    write(f'{symbol}_iso.pwi', iso_atom, input_data=input_data, kpts=kpts, pseudopotentials=pseudos)
    
    return

def input_eos(symbol, alat_0, pseudo_dir, parameters, range_perc=(-3,3), npoints=10):

    k       = parameters.get('kpts')
    offset  = parameters.get('koffset', 0) 
    pseudos = parameters.get('pseudos')

    alats = np.linspace(alat_0*(1.+range_perc[0]/100.), alat_0*(1.+range_perc[1]/100.), npoints)

    for a in alats:
        bulkfcc = bulk(symbol, 'fcc', a, cubic=True)
        input_data = input_from_yaml(parameters, len(bulkfcc), pseudo_dir, mixing_mode='TF')
        write(f'{symbol}_fcc_{a:.3f}.pwi', bulkfcc, input_data=input_data, kpts=[k,k,k], pseudopotentials=pseudos, koffset = offset)

    return

def input_fcc_relax(symbol, alat_0, pseudo_dir, parameters):

    k       = parameters.get('kpts')
    offset  = parameters.get('offset', 0)
    pseudos = parameters.get('pseudos')

    bulkfcc = bulk(symbol, 'fcc', alat_0, cubic=True)

    nat = len(bulkfcc)

    input_data = input_from_yaml(parameters, nat, pseudo_dir, calc_type='vc-relax', mixing_mode='TF')

    write(f'{symbol}_fcc_relax.pwi', bulkfcc, input_data=input_data, kpts=[k,k,k], pseudopotentials=pseudos, koffset=offset)

    return

def kpts_equiv(length, kpts_ref, length_ref, floor=True):

    k = kpts_ref*length_ref/length

    if floor==False:
        k +=1

    return k

def kpts_surf_calculator(cell, kpts_equivalent_conventional, a_ref, floor=True):

    Lx = cell[0][0]
    Ly = cell[1][1]

    kx = kpts_equiv(Lx, kpts_equivalent_conventional, a_ref, floor)
    ky = kpts_equiv(Ly, kpts_equivalent_conventional, a_ref, floor)

    return [kx, ky, 1]

def input_surfaces(symbol, pseudo_dir, vacuum, parameters, size=(1,1,8), relax=True, relax_layers=2 ):
    #writes quantum espresso inputs for relax calculations of surfaces
    #fcc 111 110 100 with dft parameters from parameters_relax.
    #Structures are created with the initial lattice constant parameters_relax[latticeconstant][symbol]
    #and kpts are rescaled to have the equivalent of parameters_relax[kpts] points in the conventional
    #fcc cell per planar direction
    #add fixed middle layers

    if relax:
        mode='relax'
    else:
        mode='scf'

    ref_lattice = parameters.get('latticeconstant')[symbol]
    kpts_equiv  = parameters.get('kpts')
    offset      = parameters.get('koffset', 0)
    pseudos     = parameters.get('pseudos')

    s111 = fcc111(symbol, size, a=ref_lattice, vacuum=vacuum)
    s110 = fcc110(symbol, size, a=ref_lattice, vacuum=vacuum)
    s100 = fcc100(symbol, size, a=ref_lattice, vacuum=vacuum)

    k111 = kpts_surf_calculator(s111.get_cell(), kpts_equiv, ref_lattice)
    k110 = kpts_surf_calculator(s110.get_cell(), kpts_equiv, ref_lattice)
    k100 = kpts_surf_calculator(s100.get_cell(), kpts_equiv, ref_lattice)

    nat = len(s111)

    #fix deep bulk atoms
    if relax:
        for surf in [s111, s110, s100]:    
            
            layers_ids = surf.get_tags()
            min_id, max_id = 1, max(layers_ids)
            min_id_fix, max_id_fix = min_id + relax_layers, max_id - relax_layers
            mask = [True if (lay_id>=min_id_fix and lay_id<=max_id_fix) else False for lay_id in layers_ids ]
            c = FixAtoms(mask=mask)
            surf.set_constraint(c)

    input_data = input_from_yaml(parameters, nat, pseudo_dir, calc_type=mode)

    write(f'{symbol}_surf_111_{size[0]}x{size[1]}x{size[2]}_relax.pwi', s111, kpts=k111, koffset=offset, pseudopotentials=pseudos, input_data=input_data)
    write(f'{symbol}_surf_110_{size[0]}x{size[1]}x{size[2]}_relax.pwi', s110, kpts=k110, koffset=offset, pseudopotentials=pseudos, input_data=input_data)
    write(f'{symbol}_surf_100_{size[0]}x{size[1]}x{size[2]}_relax.pwi', s100, kpts=k100, koffset=offset, pseudopotentials=pseudos, input_data=input_data)    

    return

def input_isomers(symbol, pseudo_dir, vacuum, parameters):
    """
    write inputs for relaxations of ih, dh, oh at 55 and 147 atoms.
    uses parameters_relax to get reference lattice constants.
    """

    ref_lattice = parameters.get('latticeconstant')[symbol]
    pseudos     = parameters.get('pseudos')


    input_data = {
        'control': {
            'calculation': 'relax',
            'pseudo_dir': pseudo_dir,
            'etot_conv_thr': parameters_relax.get('etot_conv_thr_eV_peratom')/ry*nat,
            'forc_conv_thr': parameters_relax.get('forc_conv_thr')
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

    input_data = input_from_yaml(parameters, len(ico), mixing_mode='local-TF')

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

    input_data = input_from_yaml(parameters, len(ico), mixing_mode='local-TF')

    write(f'{symbol}_Ih_{len(ico)}_relax.pwi',  ico,  input_data=input_data, kpts=None, pseudopotentials=pseudos)
    write(f'{symbol}_Oh_{len(octa)}_relax.pwi', octa, input_data=input_data, kpts=None, pseudopotentials=pseudos)
    write(f'{symbol}_Dh_{len(deca)}_relax.pwi', deca, input_data=input_data, kpts=None, pseudopotentials=pseudos)

    return

def input_dimers(symbol, separation_range, npoints, pseudo_dir, vacuum, parameters, smearing_divider=1.0):

    kpts    = None
    pseudos = parameters.get('pseudos') 
    nat     = 2

    input_data = input_from_yaml(parameters, nat, pseudo_dir, electron_maxstep=1500, mixing_mode='local-TF')

    input_data['electrons']['mixing_beta'] = 0.05

    #generate structures
    for d in np.linspace(separation_range[0], separation_range[1], npoints):

        dimer = Atoms([symbol]*2, [[0.,0.,0.], [0.,0.,d]])
        dimer.center(vacuum = vacuum) 

        write(f'{symbol}_dimer_{d:.3f}.pwi', dimer, input_data=input_data, kpts=kpts, pseudopotentials=pseudos)
    
    return

def input_phonons(symbol, 
                  alat_relax, 
                  pseudo_dir, 
                  parameters, 
                  conv_thr=1e-10, 
                  displacement_distance=0.02, 
                  supercell_size=4, 
                  files_prefix='phonopy_structure', #.pwi
                  phonon_file='phonopy' #.yaml
                  ):

    """
    use phonopy to write inputs for the calcualtions needed to get the phonon dispersion curve with 
    the finite displacement method.
    Some paramters to take extra care about:
     - conv_thr: generally needs to be tighter than usual, default to 1e-10
     - displacement_distance: the distance atoms should be displaced by in phonopy-generated configs
    Currently only working for single-specie fcc materials. Filenames get the chemical symbol appended before them.
    """

    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    #structures generation (eventually put in another function)
    #make an ase structure
    cell = crystal(symbol, [(0.,0.,0.)], spacegroup=225, cellpar=[alat_relax, alat_relax, alat_relax, 90., 90., 90.], primitive_cell=True)

    #phonopy stuff 
    phcell = PhonopyAtoms(cell=cell.get_cell(), positions=cell.get_positions(), numbers=cell.get_atomic_numbers())
    phonon = Phonopy(phcell, supercell_matrix=np.eye(3)*supercell_size)
    phonon.generate_displacements(distance=displacement_distance)

    input_data = {
        'control': {
            'calculation': 'scf',
            'pseudo_dir': pseudo_dir,
            'tprnfor': True,
            'tstress': False
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
            'conv_thr': conv_thr
        },
    }

    pseudos = parameters.get('pseudos')
    k_conv_cell = parameters.get('kpts')
    koff = parameters.get('koffset', 0)

    for i, supercell in enumerate(phonon.supercells_with_displacements):

        atoms = Atoms(cell=supercell.cell, numbers=supercell.numbers, positions=supercell.positions, pbc=True)

        k = kpts_equiv(atoms.cell.lengths()[0], k_conv_cell, alat_relax)

        write(f'{symbol}_{files_prefix}_{i}.pwi', atoms, input_data=input_data, kpts=[k,k,k], pseudopotentials=pseudos, koffset=koff)
    
    phonon.save(f'{symbol}_{phonon_file}.yaml')

    return


#########################
### parsing functions ###
#########################

def parse_qe_results(symbol, E_iso_ry=None, directory="."):
    """
    Parse Quantum ESPRESSO output files and compute:
    - E_iso: energy of isolated atom
    - fcc_lattice_constant: equilibrium lattice constant from EOS fit
    - cohesive_energy: cohesive energy per atom
    - Bulk_modulus: from EOS fit
    - 111/110/100 surface energies (for all layer configurations found)
    
    Writes output files:
    - {symbol}_eos.dat: lattice constant vs energy
    - {symbol}_dimer_dft.dat: dimer separation vs energy
    - {symbol}_{miller}_layers.dat: number of layers vs surface energy for each surface type
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
    if len(volumes)>0:
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
    else:
        print('No eos configurations found! skipping eos fitting...')

    # parse data from bulk cell relaxation
    # Use the relaxed bulk as reference
    try:
        bulk_relax = read(os.path.join(directory, f"{symbol}_fcc_relax.pwo"))
        N_bulk = len(bulk_relax)
        E_bulk = bulk_relax.get_potential_energy()
        E_coh = (E_bulk / N_bulk) - E_iso
        results["cohesive_energy"] = float(E_coh)
        results["ecoh_relax"] = float(E_coh)
        results["a0_relax"] = float( bulk_relax.get_cell().volume ** (1./3.)   )
    except Exception as e:
        print(f'Could not get data for the bulk relaxation calculation: {e}')

    # --- Surface energies ---
    # Surface energy = (E_slab - N_slab * E_bulk_per_atom) / (2 * A)
    # Factor 2 because slab has two surfaces
    E_bulk_per_atom = E_bulk / N_bulk

    # Compute surface energies for multiple layer configurations
    for miller in ["111", "110", "100"]:
        surface_files = sorted(glob.glob(os.path.join(directory, f"{symbol}_surf_{miller}_*.pwo")))
        layers_list = []
        energies_list = []
        
        for fname in surface_files:
            try:
                # Extract number of layers from filename
                # Pattern: {symbol}_surf_{miller}_{nx}x{ny}x{nlayers}_relax.pwo
                basename = os.path.basename(fname)
                match = re.search(r'_(\d+)x(\d+)x(\d+)_relax\.pwo', basename)
                if match:
                    nx, ny, nlayers = map(int, match.groups())
                else:
                    print(f'Warning: Could not extract layer count from {basename}')
                    continue
                
                slab = read(fname)
                N_slab = len(slab)
                E_slab = slab.get_potential_energy()
                cell = slab.get_cell()
                # Surface area from cross product of two in-plane lattice vectors
                A = np.linalg.norm(np.cross(cell[0], cell[1]))
                E_surf = (E_slab - N_slab * E_bulk_per_atom) / (2 * A)
                # Convert eV/Å^2 to J/m^2
                E_surf_Jm2 = E_surf * 16.0218
                
                layers_list.append(int(nlayers))
                energies_list.append(E_surf_Jm2)
                
            except Exception as e:
                print(f'Could not read {fname}: {e}')
        
        # Save surface energy data if any files were found
        if layers_list:
            # Sort by number of layers
            sorted_data = sorted(zip(layers_list, energies_list), key=lambda x: x[0])
            layers_list, energies_list = zip(*sorted_data)
            
            # Write to file
            outfile = f'{symbol}_{miller}_layers.dat'
            np.savetxt(outfile, np.column_stack((layers_list, energies_list)))
            results[f"{miller}_surface_energy_file"] = outfile
            
            # Also store average for quick reference
            avg_energy = np.mean(energies_list)
            results[f"{miller}_surface_energy"] = float(avg_energy)

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

def parse_phonons(files, phonon_file_in, phonon_file_out='phonopy_params.yaml', band_file_out='band.yaml'):

    from phonopy import Phonopy, load

    phonon = load(phonon_file_in)
    forces = []

    for f in files:
        atoms = read(f)
        forces.append(atoms.get_forces())
        
    phonon.forces = forces
    phonon.produce_force_constants()
    phonon.save(phonon_file_out, settings={"force_constants": True})
    phonon.auto_band_structure(write_yaml=True, filename=band_file_out) #to plot: phonopy-bandplot bands.yaml

    return


#####################
### main wrappers ###
#####################

def write_all_inputs(args): #args=sys.argv

    if len(args)<4:
        sys.exit(f'usage: {args} chemical_symbol vacuum pseudo_dir')

    with open('parameters.yml', 'r') as f:
        parameters = yaml.safe_load(f)

    with open('parameters_relax.yml', 'r') as f:
        parameters_relax = yaml.safe_load(f)

    symbol     = args[2] #chemical symbol
    vacuum     = float(args[3]) #vacuum for surfs
    pseudo_dir = args[4] #pseudopotentials directory 
    

    alat_0     = parameters.get('latticeconstant')[symbol]

    input_iso(symbol, pseudo_dir, parameters, smearing_divider=100.)
    input_eos(symbol, alat_0, pseudo_dir, parameters)
    input_fcc_relax(symbol, alat_0, pseudo_dir, parameters_relax)
    input_surfaces(symbol, pseudo_dir, vacuum, parameters_relax)
    input_isomers(symbol, pseudo_dir, vacuum, parameters_relax)
    input_dimers(symbol, separation_range=(alat_0/1.412*0.65, alat_0/1.412*3.0), npoints=10, pseudo_dir=pseudo_dir, vacuum=vacuum, parameters=parameters, smearing_divider=100.)

def plot_references(symbol, in_path=None, directory='.'):
    """
    A function to plot all kinds of reference values in one place after running parse_qe_results and parse_phonons
    """

    import matplotlib.pyplot as plt

    #First, load all data

    #get parsed data
    if in_path is None:
        in_path = f"{symbol}_reference_data.yaml"

    with open(in_path, 'r') as f:
        parsed_data = yaml.safe_load(f)

    #EOS
    try:
        alats_eos, energies_eos = np.loadtxt(parsed_data.get('eos_file'), unpack=True)
        eos=True
    except Exception as e:
        print(f'Did not work with {symbol}_eos.dat: {e}')
        eos=False
    
    #surface energies vs layers curves
    try:
        layers_111, energies_111 = np.loadtxt(parsed_data.get("111_surface_energy_file"), unpack=True)
        layers_110, energies_110 = np.loadtxt(parsed_data.get("110_surface_energy_file"), unpack=True)
        layers_100, energies_100 = np.loadtxt(parsed_data.get("100_surface_energy_file"), unpack=True)
        surf = True
    except Exception as e:
        print(f'Did not work with {parsed_data.get("111_surface_energy_file")} & co.: {e}')
        surf = False

    #plot
    fig, axs = plt.subplots(2,2)

    #eos
    if eos:
        axs[0,0].plot(alats_eos, energies_eos)
        axs[0,0].set_title('Equation of state')
        eos_text = f"ecoh: {parsed_data.get('cohesive_energy'):.3d} eV\nalat: {parsed_data.get('a0_eos'):.3d} Ang.\n   B: {parsed_data.get('Bulk_modulus'):.3d} GPa."
        x_text = (alats_eos[0] + alats_eos[-1])/2.
        y_text = energies_eos[0]
        axs[0,0].text(x_text,y_text,eos_text)

    if surf:
        axs[0,1].plot(layers_100, energies_100, label='100')
        axs[0,1].plot(layers_110, energies_110, label='110')
        axs[0,1].plot(layers_111, energies_111, label='111')

        axs[0,1].legend()

        axs[0,1].set_title('Surface energies')
    
    plt.show()
    

#################
### main code ###
#################

if __name__=='__main__':

    if len(sys.argv)<2 or (not (sys.argv[1]=='input' or sys.argv[1]=='parse' or sys.argv[1] == 'plot')):
        print(f'USAGE: {sys.argv[0]} <mode> <args>')
        print('--------------------------')
        print('if <mode>==input, args should be: <chemical_symbol> <vacuum_for_surfaces_and_clusters> <pseudopotenials_directory>')
        print('and parameters.yml and parameters_relax.yml should be present in the execution folder.')
        print('--------------------------')
        print('if <mode>==parse, args should be: <chemical_symbol> [isolated_atom_energy_in_Rydberg - optional]')
        print('--------------------------')
        print('if <mode> == "plot", args should be: <chemical_symbol>')

    if sys.argv[1] == 'input':
        write_all_inputs(sys.argv)
    
    elif sys.argv[1] == 'parse':
        symbol = sys.argv[2]
        if len(sys.argv)>3:
            parse_qe_results(symbol, E_iso_ry=float(sys.argv[3]) )
        else:
            parse_qe_results(symbol)
        
        parse_phonons(sorted(glob.glob(f"{symbol}_phonopy_structure_*.pwo")), f"{symbol}_phonopy.yaml")

    elif sys.argv[1] == 'plot': #deprecated
        symbol = sys.argv[2]
        plot_references(symbol)