import yaml
import numpy as np
import sys

from ase import Atoms, Atom
from ase.units import kJ, fs
from ase.lattice.cubic import FaceCenteredCubic as fcc,BodyCenteredCubic as bcc
from ase.build import bulk,add_vacuum,fcc100,fcc111,fcc110, add_adsorbate#,surface as asesurf
from ase.cluster import Icosahedron as ih, Octahedron as oh, Decahedron as dh
from ase.calculators.lammpslib import LAMMPSlib #ideally, for all potentials, in fact, use direct calculators
from ase.io import write, read
from ase.eos import EquationOfState as eqos,  calculate_eos
from ase.optimize import LBFGS
import time
from ase.md.langevin import Langevin
try:
    from lammps import lammps as lmp
except:
    lammps_ok = False
    print("Warning: lammps-python not available.")

#deal with different calculators - mace flare nequip
#maybe do a run_all script
#add output folder
#fix logging errors.txt
#try su stresses
#make a plotter with everything
#divide eos fit eos test

#flare calc
try:
    from flare.bffs.gp.calculator import FLARE_Calculator as flare_calc #needs FLARE_Atoms objects
except:
    flare_ok = False
    print("Warning: FLARE_Calculator not available.")

#mace calc
try:
    from mace.calculators import MACECalculator
except:
    mace_ok = False
    print("Warning: MACECalculator not available.")


import re

try:
    from tqdm import tqdm
except:
    def tqdm(iterable, desc=""):
        return iterable

#BENCHMARK 1: EQUATION OF STATE FOR BCC AND FCC

def eos_fcc_benchmark(symbol, calc, alat, pmppercent=3.0):
    """
    Compute bulk energies and fit the equation of state. Initial lattice constant range is alat(1-pmpercent/100), alat(1+pmpercent/100); a more precise calculation follows.
    """

    bulk_properties = dict({})
    bulk_factories  = dict({"fcc" : fcc})
    bounds_dict     = dict({ "fcc" : [alat-pmppercent*alat/100., alat+pmppercent/100]}) # a tentative alats range. more precise calculation follows after a first rough estimate
    typelats        = ["fcc"]

    #mace learns and returns E_iso, flare has it =0
    iso_atom = Atoms([symbol],[[0.,0.,0.]], pbc=False)
    iso_atom.calc = calc
    iso_atom.center(vacuum=10.0) #needed for lammps

    #iterate over lattice types - only fcc as of now
    for typelat in typelats:
        volumes, energies = [], []
        bounds            = bounds_dict[typelat]
        alats             = np.linspace(bounds[0] , bounds[1], 15)

        #compute energy over a range of latice constants around the putative minimum
        for alat in alats:
            sys      = bulk_factories[typelat](size=(1,1,1), latticeconstant=alat, symbol=symbol, pbc=(1,1,1)) 
            sys.calc = calc
            poten       = sys.get_potential_energy()
            vol         = sys.get_volume()
            energies.append(poten)
            volumes.append(vol)
            #print("calculated system with alat=",alat)

        #fit equation of state
        v0, e0, B = eqos(volumes,energies,eos='murnaghan').fit()

        #now, more precise:
        volumes, energies = [], []
        alat_precise = (v0/len(sys)*4. )**(1.0 / 3.0)
        bounds            = [alat_precise - pmppercent*alat_precise/100., alat_precise+pmppercent*alat_precise/100.]
        alats             = np.linspace(bounds[0] , bounds[1], 20)

        for alat in alats:
            sys      = bulk_factories[typelat](size=(1,1,1), latticeconstant=alat, symbol=symbol, pbc=(1,1,1)) 
            sys.calc = calc
            poten       = sys.get_potential_energy()
            vol         = sys.get_volume()
            energies.append(poten)
            volumes.append(vol)

        #fit equation of state
        v0, e0, B = eqos(volumes,energies,eos='murnaghan').fit()

        B  = (B/kJ * 1.0e24)      #bulk modulus - eV/A**3 to GPa conversion
        a0 = (v0/len(sys)*4. )**(1 / 3.0)  #lattice constant from the volume (normalize per conventional cells)
        e0 = e0/len(sys) - iso_atom.get_potential_energy()    #potential energy per atom
        bulk_properties[typelat+"_bulk"]       = dict({})
        bulk_properties[typelat+"_bulk"]['B']  = float(B)
        bulk_properties[typelat+"_bulk"]['a0'] = float(a0)
        bulk_properties[typelat+"_bulk"]['e0'] = float(e0)

        #write eos data
        f = open('eos.dat','w')
        for a, e in zip(alats, energies):
            f.write(str(a)+" "+str(e/len(sys))+" \n")
        f.close()

    return bulk_properties

def low_index_surfen(symbol, calc, ecohesive, lattice_constant):
    """
    Compute (fcc) surface energy from relaxation calculations.
    """

    surface_properties = dict({})
    surfactories=dict({"fcc100":fcc100,
                     "fcc110":fcc110,
                     "fcc111":fcc111})
    
    #mace learns and returns E_iso, flare has it =0
    iso_atom = Atoms([symbol],[[0.,0.,0.]], pbc=False)
    iso_atom.calc = calc
    iso_atom.center(vacuum=10.0)

    for surfact, surfname in zip(surfactories.values(), surfactories.keys()):

        #create structure
        surf = surfact(symbol=symbol, size=(2,2,7), a = lattice_constant)
        surf.center(vacuum=6.5,axis=2)

        #initialize calculator
        surf.calc = calc

        #relax the structure
        dyn       = LBFGS(surf,logfile="bfgs.log")
        dyn.run(fmax=1e-5)

        #compute surface energy
        cell      = surf.get_cell()
        surface   = np.linalg.norm(np.cross(cell[0],cell[1]))
        poten     = surf.get_potential_energy() - iso_atom.get_potential_energy()*len(surf)
        surfen    = (poten - ecohesive*len(surf))/2./surface*16.02
        surface_properties[surfname]=dict({"gamma":float(surfen)})

    return surface_properties

def eos_large(symbol, calc, alat, pmppercent=50.):
    """
    Only computes and returns volume vs energy. Can be used to look at performance over a wide ragne of lattice constants to look at what happens when youre far from the minimum
    (e.g. to check for ghost holes in your potential)
    """

    bulk_properties = dict({})
    bulk_factories  = dict({"fcc" : fcc})
    bounds_dict     = dict({ "fcc" : [alat-pmppercent*alat/100., alat+pmppercent/100]}) 
    typelats        = ["fcc"]

    #iterate over lattice types - only fcc as of now
    for typelat in typelats:
        volumes, energies = [], []
        bounds            = bounds_dict[typelat]
        alats             = np.linspace(bounds[0] , bounds[1], 40)

        #compute energy over a range of latice constants around the putative minimum
        for alat in alats:
            Cu_sys      = bulk_factories[typelat](size=(1,1,1), latticeconstant=alat, symbol=symbol, pbc=(1,1,1)) 
            Cu_sys.calc = calc
            poten       = Cu_sys.get_potential_energy()
            vol         = Cu_sys.get_volume()
            energies.append(poten)
            volumes.append(vol)

    return volumes, energies

def adsorbate_curve(symbol, calc, npoints=40):
    #compute the energy and forces on a particle close to a surface at different distances
    distances = np.linspace(0.8, 8.0, npoints)
    energies  = []

    for d in distances:

        #create system: surface with adsorbate
        slab = fcc111(symbol, size = (4,4,7), vacuum = 10.0)
        add_adsorbate(slab, symbol, d, 'ontop') #should be in fcc111
        slab.center(vacuum=10.0, axis=2)

        #calculators and stuff
        slab.calc   = calc
        energies.append( slab.get_potential_energy() )
    
    return distances, energies

def dimer_curve(symbol, calc, npoints=40):
    #compute the energy and forces in a dimer molecule at different distances
    distances = np.linspace(1.0, 7.0, npoints)
    energies  = []

    for d in distances:
        dimer = Atoms(symbol+symbol, positions=[(0,0,0),(0,0,d)])
        dimer.center(vacuum=10.0)

        #calculators and stuff
        dimer.calc   = calc
        energies.append( dimer.get_potential_energy() )
        #print(f'I just did {d}')

    return distances, energies

def mae_mav_test(calc, test_set_file, E_iso, use_norm=True):

    """
    Compute mae/mav on a test set. NB: energies are per atom.

    test_set_file: a string with your xyzs test set configurations
    E_iso: energy for the isolated atom (removed from dft results). To be implemented: more than one specie
    use_norm: compute errors on norm of forces and stresses (invariant wrt rotations) VS on the single components (use_norm=False)
    returns mae and mav of energy per atom, force, stress
   """

    test_set = read(test_set_file, index=':')

    e_at_mae, f_mae, s_mae = 0., 0., 0.
    e_at_mav, f_mav, s_mav = 0., 0., 0.

    out_folder = "./" #eventually set?
    errors_file = open(out_folder+'errors.dat','w') #to check if any particular configuration contributes much to the average error
    parity_e, parity_f, parity_s = open(out_folder+'e-parity.dat','w'), open(out_folder+'f-parity.dat','w'), open(out_folder+'s-parity.dat','w')

    errors_file.write('# conf_id e_mae e_mav f_mae f_mav s_mae s_mav\n')

    #mace learns and returns E_iso, flare has it =0
    iso_atom = Atoms([symbol],[[0.,0.,0.]], pbc=False)
    iso_atom.calc = calc
    iso_atom.center(vacuum=10.0)
    E_iso_model = iso_atom.get_potential_energy()

    for itconf, conf in enumerate(tqdm(test_set, desc="computing test set predictions...")):

        #store dft values
        e_at_dft = conf.get_potential_energy()/float(len(conf)) - E_iso
        f_dft = conf.get_forces()
        s_dft = conf.get_stress()

        #compute values with new calculator
        conf.calc = calc

        #store calc values
        e_at_model = conf.get_potential_energy()/float(len(conf)) - E_iso_model
        f_model = conf.get_forces()
        s_model = conf.get_stress()

        #compute mavs (DFT)
        e_at_mav += np.abs(e_at_dft)
        if use_norm:
            f_dft_norms = [np.linalg.norm(f) for f in f_dft]
            f_model_norms = [np.linalg.norm(f) for f in f_model]
            s_dft_norm = np.linalg.norm(s_dft)
            s_model_norm = np.linalg.norm(s_model)
            f_mav += np.average(f_dft_norms)
            s_mav += s_dft_norm
        else:
            f_mav += np.average( np.ravel( np.abs(f_dft )))
            s_mav += np.average( np.ravel( np.abs(s_dft )))

        #compute maes
        e_at_mae += np.abs(e_at_dft-e_at_model)
        if use_norm:
            f_dist_norms = [np.linalg.norm(f_d-f_f) for f_d, f_f in zip(f_dft, f_model)]
            s_dist_norm  = np.linalg.norm( s_dft-s_model ) #Frobenius norm
            f_mae += np.average(f_dist_norms) #2-norm
            s_mae += s_dist_norm
        else:
            f_mae += np.average( np.ravel( np.abs(f_dft - f_model) )) 
            s_mae += np.average( np.ravel( np.abs(s_dft - s_model) ))

        #write to output - add use_norm condition
        errors_file.write(
            f"{itconf} "
            f"{np.abs(e_at_dft - e_at_model):.3g} "
            f"{np.abs(e_at_dft):.3g} "
            f"{np.average(f_dist_norms):.3g} "
            f"{np.average(f_dft_norms):.3g} "
            f"{s_dist_norm:.3g} "
            f"{s_dft_norm:.3g}\n"
        )

        parity_e.write(f"{e_at_dft} {e_at_model}\n")
        for fd, fm in zip(f_dft_norms, f_model_norms):
            parity_f.write(f"{fd} {fm}\n")
        parity_s.write(f"{s_dft_norm} {s_model_norm}\n")


    errors_file.close()
    parity_e.close()
    parity_f.close()
    parity_s.close()



    nconf = float(len(test_set))

    e_at_mav /= nconf
    f_mav /= nconf
    s_mav /= nconf

    e_at_mae  /= nconf
    f_mae  /= nconf
    s_mae  /= nconf

    return e_at_mae, e_at_mav, f_mae, f_mav, s_mae, s_mav

def clusters_excess_energy(symbol, calc, alat, ecoh, max_size=800, f_thresh=1e-7):
    """
    Compute excess energy for a range of clusters sizes and structures.
    Can't be done parallel! So it's pretty slow. probably ok with gpus.
    """

    icos = []
    octas = []
    decas = []

    #####################
    #generate structures#
    #####################

    ico_iter = 2
    curr_ico = ih(symbol, ico_iter, alat)
    while(len(curr_ico)<max_size):
        icos.append(curr_ico)
        ico_iter +=1
        curr_ico = ih(symbol, ico_iter, alat)
    
    deca_iter = 2
    curr_deca = dh(symbol, deca_iter, deca_iter, 0, alat)
    while(len(curr_deca)<max_size):
        decas.append(curr_deca)
        for i in [-1, 0, 1]:
            for j in range(2):
                curr_deca = dh(symbol, deca_iter, deca_iter+i, j, alat)
                if(len(curr_deca)<max_size):
                    decas.append(curr_deca)
        deca_iter +=1
        curr_deca = dh(symbol, deca_iter, deca_iter, 0, alat)
    
    octa_iter = 3
    curr_octa = oh(symbol, octa_iter, 1, alat) #regular truncated octahedron: l = 2*cut+1 
    while(len(curr_octa)<max_size):
        octas.append(curr_octa)
        curr_octa = oh(symbol, octa_iter, int((octa_iter-1)/3), alat) #cuboctahedron: l = 3*cut + 1 (not always possible)
        if(len(curr_octa)<max_size):
            octas.append(curr_octa)
        octa_iter +=1
        curr_octa = oh(symbol, octa_iter, int((octa_iter-1)/2), alat)

    ico_sizes  = [len(ico) for ico in icos]
    octa_sizes  = [len(octa) for octa in octas]
    deca_sizes = [len(deca) for deca in decas]

    print('Built structures:', len(icos), 'icos,', len(octas), 'octas,',len(decas),'decas')
    print(f'Max ico  size: {max(ico_sizes)}')
    print(f'Max octa size: {max(octa_sizes)}')
    print(f'Max deca size: {max(deca_sizes)}')

    geometries = [icos, octas, decas]
    names = ['ico', 'octa', 'deca']

    #####################
    #Optimize structures#
    #####################

    #mace learns and returns E_iso, flare has it =0
    iso_atom = Atoms([symbol],[[0.,0.,0.]], pbc=False)
    iso_atom.calc = calc
    iso_atom.center(vacuum=10.0)
    E_iso_model = iso_atom.get_potential_energy()

    for geom, name in zip(geometries, names):

        print('minimizing',name+'s')
        stream = open(name+'-exc.dat', 'w')

        for c in tqdm(geom, desc="warning - this won't be linear"):

            c.center(vacuum=6.5)
            c.calc = calc

            #relax the structure
            dyn = LBFGS(c, logfile="bfgs.log")
            dyn.run(fmax=f_thresh)

            #compute excess energy
            N = len(c)
            excess = (c.get_potential_energy() - N*(ecoh+E_iso_model))/(N**(2./3.))
            stream.write(f"{N} {excess}\n")

        stream.close()

    return

def perc_diff(reference, value):
    return (value-reference)/reference*100.

def MD_performance(atoms, calc, steps=1000, temperature_K=1000):
    """
    Run a few MD steps to compute the performance (katom step/s) (NB on one processor with ASE).
    leave calc=none if your atoms object already have a calculator attached to it.
    """

    atoms.calc = calc
    atoms.center(vacuum=6.5)
    dyn = Langevin(atoms, 0.5*fs, temperature_K=temperature_K, friction=1e-2)

    tic = time.time()
    dyn.run(steps)
    run_time = time.time()-tic
    
    print(f'finished running MD in {run_time/60.} minutes')

    return len(atoms)*steps/run_time

if __name__ == '__main__':

    if len(sys.argv)<1:
        print('usage:',sys.argv[0],' <setup_file>')

    #load config.yml settings 
    with open(sys.argv[1],'r') as f:
        setup = yaml.safe_load(f)

    #chemical symbol - single specie only (for now)
    symbol = setup['symbol']

    #dft references
    E_iso = setup['E_iso']

    a_ref = setup['fcc_lattice_constant']
    e_ref = setup['cohesive_energy']
    B_ref = setup['Bulk_modulus']

    dft111 = setup['111_surface_energy']
    dft110 = setup['110_surface_energy']
    dft100 = setup['100_surface_energy']

    test_set_file=setup['test_set_file']

    #init calculator
    if setup["calculator"] == "flare_lammps":

        #lammps+flare commands
        cmds= ["pair_style flare",
            "pair_coeff * * "+setup["model_file"]]

        #a common calculator for the entire program - some things don't work otherwise
        calc = LAMMPSlib(lmpcmds=cmds, log_file="test.log", keep_alive=True)
    
    elif setup["calculator"] == "flare":
        exit("not implemented yet")

    elif setup["calculator"] == "mace":
        calc = MACECalculator(model_path=setup["model_file"], device='cpu') #hard-coded device for now
    elif setup["calculator"] == "mace_mp":
        from mace.calculators import mace_mp
        calc = mace_mp()

    elif setup["calculator"] == "nequip":
        exit("not implemented yet")

    #compute bulk&surface values
    print('computing dft properties')
    bulk_properties = eos_fcc_benchmark(symbol, calc, a_ref)
    surf_properties = low_index_surfen(symbol, calc, bulk_properties['fcc_bulk']['e0'], bulk_properties['fcc_bulk']['a0'])
    properties = {**surf_properties, **bulk_properties}

    #compare with known dft values, compute and store percentage errors
    a_p = perc_diff(a_ref, properties['fcc_bulk']['a0'])
    e_p = perc_diff(e_ref, properties['fcc_bulk']['e0'])
    B_p = perc_diff(B_ref, properties['fcc_bulk']['B'])
    g1_p = perc_diff(dft111, properties['fcc111']['gamma'])
    g2_p = perc_diff(dft110, properties['fcc110']['gamma'])
    g3_p = perc_diff(dft100, properties['fcc100']['gamma'])

    properties['fcc_bulk']['a0_rel_error'] = a_p
    properties['fcc_bulk']['e0_rel_error'] = e_p
    properties['fcc_bulk']['B_rel_error']  = B_p
    properties['fcc111']['rel_error'] = g1_p
    properties['fcc110']['rel_error'] = g2_p
    properties['fcc100']['rel_error'] = g3_p        

    #MAE, MAV on test set
    print('computing test set errors')
    e_mae, e_mav, f_mae, f_mav, s_mae, s_mav = mae_mav_test(calc, test_set_file, E_iso)

    properties['test_set'] = {
        'energy_per_atom': {},
        'forces': {},
        'stress': {}
        }
    
    properties['test_set']['energy_per_atom']['mae'] = float(e_mae)
    properties['test_set']['energy_per_atom']['mav'] = float(e_mav)
    properties['test_set']['energy_per_atom']['ratio_x100'] = float(e_mae/e_mav*100)

    properties['test_set']['forces']['mae'] = float(f_mae)
    properties['test_set']['forces']['mav'] = float(f_mav)
    properties['test_set']['forces']['ratio_x100'] = float(f_mae/f_mav*100)

    properties['test_set']['stress']['mae'] = float(s_mae)
    properties['test_set']['stress']['mav'] = float(s_mav)
    properties['test_set']['stress']['ratio_x100'] = float(s_mae/s_mav*100)

    #COMPUTE MD Computational PERFORMANCE
    print('computing performance')
    ico = ih(symbol, 4, a_ref)
    properties["performance_atom_step_s"] = MD_performance(ico, calc, steps=100)

    print(yaml.dump(properties, sort_keys=False, default_flow_style=False, indent=4))
    #save to file
    f = open(setup["calculator"]+"_benchmark.yaml",'w')
    yaml.dump(properties, f)
    f.close()

    #ADSORBATE/DIMER: CHECK FOR INSTABILITIES
    print('computing distant atom curves')
    d, e = adsorbate_curve(symbol,calc)
    f = open('adsorbate_curve.dat','w')
    for dd, ee in zip(d, e):
        f.write(str(dd)+' '+str(ee)+'\n')
    f.close()

    d, e = dimer_curve(symbol,calc)
    f = open('dimer_curve.dat','w')
    for dd, ee in zip(d, e):
        f.write(str(dd)+' '+str(ee)+'\n')
    f.close()

    #EXCESS ENERGIES
    print('computing excess energies')
    clusters_excess_energy(symbol,calc, properties['fcc_bulk']['a0'], properties['fcc_bulk']['e0'], max_size=500)

    print("Done.")
