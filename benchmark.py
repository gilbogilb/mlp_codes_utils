import yaml
import numpy as np
import sys

from ase import Atoms,data
from ase.units import kJ
from ase.lattice.cubic import FaceCenteredCubic as fcc,BodyCenteredCubic as bcc
from ase.build import bulk,add_vacuum,fcc100,fcc111,fcc110, add_adsorbate#,surface as asesurf
from ase.cluster import Icosahedron as ih, Octahedron as oh, Decahedron as dh
from ase.calculators.lammpslib import LAMMPSlib
from ase.io import write, read
from ase.eos import EquationOfState as eqos,  calculate_eos
from ase.optimize import LBFGS

from lammps import lammps as lmp

import re

try:
    from tqdm import tqdm
except:
    def tqdm(iterable):
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

    #iterate over lattice types - only fcc as of now
    for typelat in typelats:
        volumes, energies = [], []
        bounds            = bounds_dict[typelat]
        alats             = np.linspace(bounds[0] , bounds[1], 15)

        #compute energy over a range of latice constants around the putative minimum
        for alat in alats:
            Cu_sys      = bulk_factories[typelat](size=(1,1,1), latticeconstant=alat, symbol=symbol, pbc=(1,1,1)) 
            Cu_sys.calc = calc
            poten       = Cu_sys.get_potential_energy()
            vol         = Cu_sys.get_volume()
            energies.append(poten)
            volumes.append(vol)
            #print("calculated system with alat=",alat)

        #fit equation of state
        v0, e0, B = eqos(volumes,energies,eos='murnaghan').fit()

        #now, more precise:
        volumes, energies = [], []
        alat_precise = (v0/len(Cu_sys)*4. )**(1.0 / 3.0)
        bounds            = [alat_precise - pmppercent*alat_precise/100., alat_precise+pmppercent*alat_precise/100.]
        alats             = np.linspace(bounds[0] , bounds[1], 20)

        for alat in alats:
            Cu_sys      = bulk_factories[typelat](size=(1,1,1), latticeconstant=alat, symbol=symbol, pbc=(1,1,1)) 
            Cu_sys.calc = calc
            poten       = Cu_sys.get_potential_energy()
            vol         = Cu_sys.get_volume()
            energies.append(poten)
            volumes.append(vol)

        #fit equation of state
        v0, e0, B = eqos(volumes,energies,eos='murnaghan').fit()

        B  = (B/kJ * 1.0e24)      #bulk modulus - eV/A**3 to GPa conversion
        a0 = (v0/len(Cu_sys)*4. )**(1 / 3.0)  #lattice constant from the volume (normalize per conventional cells)
        e0 = e0/len(Cu_sys)    #potential energy per atom
        bulk_properties[typelat+"_bulk"]       = dict({})
        bulk_properties[typelat+"_bulk"]['B']  = float(B)
        bulk_properties[typelat+"_bulk"]['a0'] = float(a0)
        bulk_properties[typelat+"_bulk"]['e0'] = float(e0)

        #write eos data
        f = open('eos.dat','w')
        for a, e in zip(alats, energies):
            f.write(str(a)+" "+str(e/len(Cu_sys))+" \n")
        f.close()
        print("For crystal structure",typelat," a0,e0,B are:",a0,e0,B)

    return bulk_properties

def low_index_surfen(symbol, calc, ecohesive, lattice_constant):
    """
    Compute surface energy from relaxation calculations.
    """

    surface_properties = dict({})
    surfactories=dict({"fcc100":fcc100,
                     "fcc110":fcc110,
                     "fcc111":fcc111})
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
        poten     = surf.get_potential_energy()
        surfen    = (poten - ecohesive*len(surf))/2./surface*16.02
        surface_properties[surfname]=dict({"gamma":float(surfen)})

    return surface_properties

def eos_large(symbol, calc, alat=3.57, pmppercent=50.):
    """
    Only computes and returns volume vs energy. Can be used to look at performance over a wide ragne of lattice constants to look at what happens when youre far from the minimum
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

def adsorbate_curve(symbol, calc, npoints=80):
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

def dimer_curve(symbol, calc, npoints=80):
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

def mae_mav_test(calc, test_set_file, E_iso):

    """
    Compute mae/mav on a test set. NB: energies are per atom. With forces, we deal with magnitudes rather than single-axis components
    """

    test_set = read(test_set_file, index=':')

    e_at_mae, f_mae, s_mae = 0., 0., 0.
    e_at_mav, f_mav, s_mav = 0., 0., 0.
    f_mae_components, f_mav_components = 0., 0.

    errors_file = open('errors.txt','w') #to check if any particular configuration contributes much to the average error

    for itconf, conf in enumerate(test_set):

        #store dft values
        e_at_dft = conf.get_potential_energy()/float(len(conf)) - E_iso
        f_dft = conf.get_forces()
        f_dft_norms = [np.linalg.norm(f) for f in f_dft]
        s_dft = conf.get_stress()

        #compute values with flare
        conf.calc = calc

        #store flare values
        e_at_flare = conf.get_potential_energy()/float(len(conf))
        f_flare = conf.get_forces()
        s_flare = conf.get_stress()
        f_dist_norm = [np.linalg.norm(f_d-f_f) for f_d, f_f in zip(f_dft, f_flare)]

        #compute mavs (DFT)
        e_at_mav += np.abs(e_at_dft)
        #f_mav += np.average( np.ravel( np.abs(f_dft )))
        #s_mav += np.average( np.ravel( np.abs(s_dft )))
        f_mav_components += np.average( np.ravel( np.abs(f_dft )))
        f_mav += np.average(f_dft_norms)
        s_mav += np.linalg.norm(s_dft)

        #compute maes
        e_at_mae += np.abs(e_at_dft-e_at_flare)
        #f_mae += np.average( np.ravel( np.abs(f_dft - f_flare) )) 
        #s_mae += np.average( np.ravel( np.abs(s_dft - s_flare) ))
        f_mae_components += np.average( np.ravel( np.abs(f_dft - f_flare) )) 
        f_mae += np.average(f_dist_norm) #2-norm
        s_mae += np.linalg.norm( s_dft-s_flare ) #Frobenius norm

        #test differences between the two methods
        errors_file.write(
            f"{itconf} "
            f"{np.abs(e_at_dft - e_at_flare):.3g} "
            f"{np.abs(e_at_dft):.3g} "
            f"{np.average(f_dist_norm):.3g} "
            f"{np.average(f_dft_norms):.3g} "
            f"{np.linalg.norm(s_dft - s_flare):.3g} "
            f"{np.linalg.norm(s_dft):.3g}\n"
        )

    errors_file.close()

    nconf = float(len(test_set))

    e_at_mav /= nconf
    f_mav /= nconf
    s_mav /= nconf

    e_at_mae  /= nconf
    f_mae  /= nconf
    s_mae  /= nconf

    return e_at_mae, e_at_mav, f_mae, f_mav, s_mae, s_mav



def clusters_excess_energy(symbol, calc, alat, ecoh, max_size=850):
    """
    Compute excess energy for a range of clusters sizes and structures.
    Can't be done parallel! So it's pretty slow
    """
    import subprocess #roncio but parallel. update in the future with python lammps + mpi4py, faster and more portable

    icos = [ih(symbol, i, alat) for i in range(2,15) if len(ih(symbol, i, alat))<max_size]
    octas = []
    decas = []
    for n in range(2,15):
        for cut in range(8):
            if cut <= (n-1)/2.:
                clust = oh(symbol, n, cut, alat)
                if len(clust)<max_size:
                    octas.append(clust)
    for m in range(2,10):
        for n in range(2,10):
            for p in range(4):
                clust = dh(symbol,m,n,p,alat)
                if(len(clust)<max_size):
                    decas.append(clust)

    print('Built structures:', len(icos), 'icos,', len(octas), 'octas,',len(decas),'decas. Optimizing...')

    ico_sizes  = [len(ico) for ico in icos]
    octa_sizes  = [len(octa) for octa in octas]
    deca_sizes = [len(deca) for deca in decas]

    print(ico_sizes, octa_sizes, deca_sizes)

    geometries = [icos, octas, decas]
    names      = ['ico', 'octa', 'deca']
    exc_ico, exc_octa, exc_deca = [], [], []
    for geom, exc in zip(geometries, [exc_ico, exc_octa, exc_deca]):
        for c in tqdm(geom):

            c.center(vacuum=10.0)

            #initialize calculator
            c.calc = calc

            #relax the structure
            dyn       = LBFGS(c, logfile="bfgs.log")
            dyn.run(fmax=1e-8) #could be that forces are zero for simmetry???

            #compute excess energy
            N = len(c)
            excess = (c.get_potential_energy() - N*ecoh)/(N**(2./3.))
            exc.append(excess)
        print('done ',geom)

    return [[ico_sizes, octa_sizes, deca_sizes], [exc_ico, exc_octa, exc_deca]]

    return 

def perc_diff(reference, value):
    return (value-reference)/reference*100.

if __name__ == '__main__':

    if len(sys.argv)<2:
        print('usage:',sys.argv[0],' <setup_file> [<potential (coefficients) file>]')

    if len(sys.argv)>2:
        flare_file=sys.argv[2]
        print('using ',sys.argv[2], 'as the potential file')
        # match anything ending in "coeffs.dat"
        m = re.match(r"(.+?)coeffs\.dat$", flare_file)
        if m:
            prefix = m.group(1)     # everything before "coeffs.dat"
            print("Detected prefix:", prefix)
        else:
            prefix = ""
    else:
        flare_file="lmp.flare"
        print('did not provide flare potential file, using default lmp.flare')
        prefix = ""
    
    if len(sys.argv)>3:
        prefix = sys.argv[3]
        print('using prefix ',prefix)

    #lammps+flare commands
    folder    = ""
    cmds= ["pair_style flare",
        "pair_coeff * * "+folder+flare_file]

    #a common calculator for the entire program - some things don't work otherwise
    lammps = LAMMPSlib(lmpcmds=cmds, log_file="test.log", keep_alive=True)

    #DFT setup and know values/benchmarks
    with open(sys.argv[1],'r') as f:
        setup = yaml.safe_load(f)

    symbol = setup['symbol']
    E_iso = setup['E_iso']

    a_dft = setup['fcc_lattice_constant']
    e_dft = setup['cohesive_energy']
    B_dft = setup['Bulk_modulus']

    dft111 = setup['111_surface_energy']
    dft110 = setup['110_surface_energy']
    dft100 = setup['100_surface_energy']

    test_set_file=setup['test_set_file']

    #compute bulk&surface values
    bulk_properties = eos_fcc_benchmark(symbol, lammps, a_dft)
    surf_properties = low_index_surfen(symbol, lammps, bulk_properties['fcc_bulk']['e0'], bulk_properties['fcc_bulk']['a0'])

    properties = {**surf_properties, **bulk_properties}

    #compare with known dft values, compute and store percentage errors
    a_p = perc_diff(a_dft, properties['fcc_bulk']['a0'])
    e_p = perc_diff(e_dft, properties['fcc_bulk']['e0'])
    B_p = perc_diff(B_dft, properties['fcc_bulk']['B'])
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
    print('computing mae, mav...')
    e_mae, e_mav, f_mae, f_mav, s_mae, s_mav = mae_mav_test(lammps, test_set_file, E_iso)

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

    #save to file
    f = open(folder+prefix+"flare_benchmark.yaml",'w')
    yaml.dump(properties, f)
    f.close()

    print(yaml.dump(properties, sort_keys=False, default_flow_style=False, indent=4))

    #ADSORBATE/DIMER: CHECK FOR INSTABILITIES
    print('computing adsorbate...')
    d, e = adsorbate_curve(symbol,lammps,40)
    f = open(folder+'adsorbate_curve.dat','w')
    for dd, ee in zip(d, e):
        f.write(str(dd)+' '+str(ee)+'\n')
    f.close()

    print('computing dimer...')
    d, e = dimer_curve(symbol,lammps,40)
    f = open(folder+'dimer_curve.dat','w')
    for dd, ee in zip(d, e):
        f.write(str(dd)+' '+str(ee)+'\n')
    f.close()

    #EXCESS ENERGIES
#    print('computing excess energies...')
#    sizes, excess_energies = clusters_excess_energy(symbol, lammps, properties['fcc_bulk']['a0'], properties['fcc_bulk']['e0'], max_size= 400) #now automatically writes to file

#    fout = open(folder+'ico-exc.dat','w')
#    for N, e in zip(sizes[0], excess_energies[0]):
#        fout.write(str(N)+' '+str(e)+'\n')
#    fout.close()

#    fout = open(folder+'octa-exc.dat','w')
#    for N, e in zip(sizes[1], excess_energies[1]):
#        fout.write(str(N)+' '+str(e)+'\n')
#    fout.close()

#    fout = open(folder+'deca-exc.dat','w')
#    for N, e in zip(sizes[2], excess_energies[2]):
#        fout.write(str(N)+' '+str(e)+'\n')
#    fout.close()


    print("Done.")
