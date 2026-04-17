
#some code by davide alimonti
#extensions: #add stress in ase2flare?
#add stress error
#take care with randomness - here, np.radnom is used, should set the seed in np as well;

import numpy as np
from scipy.optimize import minimize
from scipy.special import huber

import sys
import time
import json
import tempfile
import random
import yaml
from datetime import datetime
from copy import deepcopy

import flare
from flare.bffs.sgp.calculator import SGP_Calculator
from flare.bffs.gp.calculator import FLARE_Calculator
from flare.bffs.sgp._C_flare import   B2, NormalizedDotProduct, SparseGP, Structure
from flare.bffs.sgp import SGP_Wrapper
from flare.learners.otf import OTF
from flare.io import otf_parser
from flare.scripts.otf_train import get_sgp_calc
from flare.atoms import FLARE_Atoms

from ase.io import read,write
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from makesets import *
from benchmark import compute_test_errors

try:
    from tqdm import tqdm
except:
    def tqdm(iterable):
        return iterable





def ase2flare(struct, config, descriptors):
    """
    Takes an ASE structure and returns a FLARE structure object
    """

    species_code = config["species_code"]
    isolated_energies = config["isolated_energies"]

    noa = len(struct.numbers)
    coded_species=[]
    eisol = 0
    for spec in struct.numbers:
        coded_species.append(species_code[str(spec)])
        eisol += isolated_energies[str(spec)]
    cell = struct.cell.array
    pos = struct.positions
    cutoff = config["flare_calc"]["cutoff"]
    flare_struct = Structure(cell, list(coded_species), pos, cutoff, descriptors)
    flare_struct.wrap_positions()
    if "forces" in struct.arrays:
        flare_struct.forces = struct.arrays["forces"].reshape(-1)
    else:
        flare_struct.forces = struct.get_forces().reshape(-1)
    flare_struct.energy = np.array([struct.calc.get_potential_energy() - eisol])
    if "stresses" in struct.arrays:
        pass
    return flare_struct

def flare_atoms_to_structure(flare_atoms):
    from flare.structure import Structure

    cell = flare_atoms.get_cell().array
    positions = flare_atoms.get_positions()
    species = flare_atoms.get_atomic_numbers()

    unique = sorted(set(species))
    mapping = {Z: i for i, Z in enumerate(unique)}
    species = [mapping[Z] for Z in species]

    struc = Structure(cell, species, positions)

    if flare_atoms.calc is not None:
        struc.energy = flare_atoms.get_potential_energy()
        struc.forces = flare_atoms.get_forces().flatten()
        try:
            struc.stresses = flare_atoms.get_stress()
        except:
            pass

    return struc


def check_mae(gp_model, train_struct):
    """
    Returns the error on energy/per atom and the array of forces errors incurred by the gp_model on a FLARE structure train_struct
    """
    force_components_errors =[]
    nat=len(train_struct.species)
    gp_model.predict_local_uncertainties(train_struct)
    energy_error = (train_struct.energy -  train_struct.mean_efs[0]) / nat
    force_components_errors = (train_struct.forces - train_struct.mean_efs[1:-6]).tolist()
    return energy_error, np.array(force_components_errors)

def log_errors(gp_model, testsets):
    """
    Measures the error incurred by the potential on a collection of testsets (made of FLARE structures)
    """
    file_maes_e.write(f"{step}\t")
    file_maes_f.write(f"{step}\t")
    for testset in testsets:
        enerrs, fcerrs = np.empty(0), np.empty(0)
        for test_struct in testset:
            enerr, fcerr = check_mae(gp_model, test_struct)
            enerrs  = np.concatenate((enerrs, enerr))
            fcerrs  = np.concatenate((fcerrs, fcerr))
        mae_e = np.mean(np.abs(enerrs))
        mae_f = np.mean(np.abs(fcerrs))
        file_maes_e.write(f"{mae_e:.5f}\t")
        file_maes_f.write(f"{mae_f:.5f}\t")
    file_maes_e.write("\n")
    file_maes_f.write("\n")
    file_maes_e.flush()
    file_maes_f.flush()
    return

#TODO: REFINE
def log_errors_gibo(errors, file_maes_e, file_maes_f, step):
    #only one test set here
    str_e = f"{step}\t{errors[0]['test.xyz']['energy_per_atom']['mae']}\n"
    str_f = f"{step}\t{errors[0]['test.xyz']['forces']['mae']}\n"
    file_maes_e.write(str_e)
    file_maes_f.write(str_f)
    file_maes_e.flush()
    file_maes_f.flush()

    return


def compute_negative_likelihood_grad_stable(
    hyperparameters, sparse_gp, precomputed=False):
    """
    Compute the negative log likelihood and gradient with respect to the
    hyperparameters.
    """

    assert len(hyperparameters) == len(sparse_gp.hyperparameters)

    sparse_gp.set_hyperparameters(hyperparameters)

    negative_likelihood = -sparse_gp.compute_likelihood_gradient_stable(precomputed)
    negative_likelihood_gradient = -sparse_gp.likelihood_gradient

    return negative_likelihood, negative_likelihood_gradient

def optimize_hyps(gp_model, 
                opt_method,
                minhyps,
                maxhyps,
                max_iterations,
                bounds,
                gtol,
                loss_function_config):

    """
    Finds optimal hyperparameters for the gp_model, using its current hyps as starting guess.
    If the found hyps are outside the defined minhyps,maxhyps, it will return True - signalling a failure
    otherwise it will set new hyps to the model
    """
    rollback = False
    initial_guess = gp_model.hyperparameters
    old_hyps      = np.array(initial_guess) # save in case of rollback
    if loss_function_config["name"] == "negative_likelihood":
        loss_function = compute_negative_likelihood_grad_stable
        arguments = (gp_model, True)
        gp_model.precompute_KnK()
        jac = True
    elif loss_function_config["name"] == "huber" :
        loss_function = huber_loss
        arguments = (gp_model,loss_function_config["weights"])
        jac = "2-points"

    optimization_result = minimize(
                loss_function,
                initial_guess,
                arguments,
                method=opt_method,
                jac=jac,
                options={
                    "disp": False,
                    "gtol": gtol ,
                    "maxiter": max_iterations,
                    "eps" : np.array([1e-3,1e-4,1e-4,1e-5])
                }
            )
    #print(optimization_result)
    # Assign likelihood gradient, if it didn't explode
    #print(f'old hyps: {old_hyps}\n optimized hyps: {optimization_result.x}')
    if np.all(np.abs(optimization_result.x) < maxhyps) and np.all(np.abs(optimization_result.x) > minhyps) :
        # Optimization succedeed, so set new hyps
        gp_model.set_hyperparameters(np.abs(optimization_result.x))
        gp_model.update_matrices_QR()
        if loss_function_config["name"] == "negative_likelihood":
            gp_model.likelihood_gradient = -optimization_result.jac
            gp_model.log_marginal_likelihood = -optimization_result.fun
    else:
        # Optimization failed. Flag this, and reset old hyps.
        gp_model.set_hyperparameters(old_hyps)
        #file_log.write("Optimization resulted in exploded or collapsed hyps!\n")
        #file_log.write(f"Would have been : {np.array2string(np.abs(optimization_result.x))}"+'\n')
        rollback = True
        #file_log.write("Hyps NOT updated\n")

    return rollback

def write_to_json(gp_model, power, radial_basis_type,
                  cutoff_function, cutoff, nspecies, nmax, lmax,
                  opt_method, variance_type, max_iterations,
                  minhyps, maxhyps,bounds,
                  sigma_e,sigma_f,sigma_s,sigma,
                  isolated_energies_mapped, descriptor_type, gtol, loss_function_config, author='user'):
    """
    Returns a JSON of the model including all necessary data to retrain it.
    It also produces the maps of the model
    """
    hyperlist=np.array(gp_model.hyperparameters).tolist()
    #log_errors(gp_model,testsets)
    gp_model.write_mapping_coefficients(f"lmp.flare",author,0)
    gp_model.write_sparse_descriptors(f"sparse_desc_lmp.flare",author)
    gp_model.write_L_inverse(f"L_inv_lmp.flare",author)
    gpmodeldict = dict({"sparse_indices": [sparse_indices], "training_structures": training_structures})
    gpmodeldict["cutoff"] = cutoff
    gpmodeldict["species_map"] = species_code
    gpmodeldict["variance_type"] = variance_type
    gpmodeldict["single_atom_energies"] = isolated_energies_mapped
    gpmodeldict["energy_training"] = True
    gpmodeldict["force_training"] = True
    gpmodeldict["stresses_training"] = False
    gpmodeldict["descriptor_calculators"] = [{'type': descriptor_type, 'radial_basis': radial_basis_type, 'cutoff_function': cutoff_function, 'radial_hyps': [0.0, cutoff], 'cutoff_hyps': [], 'descriptor_settings': [power, nmax, lmax], 'cutoffs': [[cutoff]]}]
    gpmodeldict["Kuu_jitter"] = 1e-8
    gpmodeldict["hyps_mask"] = None
    gpmodeldict["max_iterations"] = max_iterations
    gpmodeldict["opt_method"] = opt_method
    gpmodeldict["bounds"] = None
    gpmodeldict["atom_indices"] = [ [-1] for _ in range(len(training_structures))]
    gpmodeldict["rel_efs_noise"] = [ [1,1,1] for _ in range(len(training_structures))]
    gpmodeldict["hyps"] = hyperlist
    gpmodeldict["kernels"] = [['NormalizedDotProduct', hyperlist[0]  , 2.0]]
    gpmodeldict["hyp_labels"] =  ['Hyp0', 'Hyp1', 'Hyp2', 'Hyp3']
    gpmodeldict["sgp_var_flag"] = "new"

    finaldict = dict(
            {
                "gp_model" : gpmodeldict,
                "results" : {},
                "parameters" : {},
                "_directory" : ".",
                "prefix" : None,
                "name" : "sgp_calculator",
                "use_mapping" : True,
                "mgp_model"   : None,
                "class"       : "SGP_Calculator"
                })


    with open(f"offline_{files_prefix}.json",'w') as f:
        json.dump(finaldict,f)
    return


def initialize_gp(
        sigma, power, radial_basis_type, cutoff_function, cutoff,
        nspecies, nmax, lmax,
        sigma_e, sigma_f, sigma_s) :
    """
    Create an empty model with working kernels.
    Also creates the kernel and descriptor objects.
    """
    kernels = [ NormalizedDotProduct(sigma , power) ]
    descriptors = [ B2(radial_basis_type,cutoff_function,[0,cutoff],[] , [nspecies , nmax, lmax]) ]
    gp_model_init = SparseGP( kernels, sigma_e , sigma_f, sigma_s)
    return gp_model_init, descriptors, kernels


def model_from_dict(structuresdict, sparse_indices, species_code, hyps, modelstruct):
    """
    Retrain a gp model from a dictionary
    """

    sigma   = hyps[0]
    sigma_e = hyps[1]
    sigma_f = hyps[2]
    sigma_s = hyps[3]
    power = modelstruct[0]
    nspecies = modelstruct[1]
    nmax = modelstruct[2]
    lmax = modelstruct[3]
    cutoff = modelstruct[4]

    kernels = [ NormalizedDotProduct(sigma , power) ]
    descriptors = [ B2("chebyshev","quadratic",[0,cutoff],[] , [nspecies , nmax, lmax]) ]
    gp_model = SparseGP(kernels, sigma_e, sigma_f ,sigma_s)

    idx=0
    alldata= len(structuresdict)
    for struct,indices in zip(structuresdict,sparse_indices):
        coded_species=[]
        energy = np.array(struct["results"]["energy"])
        for n in struct["numbers"]:
            coded_species.append(species_code[str(n)])
            energy[0] -= isolated_energies[str(n)]
        flare_structure = Structure(struct["cell"],coded_species,struct["positions"],cutoff,descriptors)
        flare_structure.forces = np.array(struct["results"]["forces"]).reshape(-1)
        flare_structure.energy = energy
        gp_model.add_training_structure(flare_structure)
        gp_model.add_specific_environments(flare_structure,indices)
        idx += 1
    return gp_model,descriptors,kernels

def ase2dict(struct: Atoms) -> dict :
    """
    Returns the dictionary with the FLARE information from ASE Atoms object struct
    """
    structdict = dict({
        "numbers"  : struct.numbers.tolist(),
        "positions": struct.positions.tolist(),
        "cell"     : struct.cell.tolist(),
        "pbc"      : struct.pbc.tolist(),
        "info"     : dict({'rel_efs_noise': [1, 1, 1]}),
        "results"  : dict({"forces": struct.arrays["forces"].tolist() if "forces" in struct.arrays else struct.get_forces().tolist() , "energy" : [struct.calc.get_potential_energy().tolist()], "stresses": struct.get_stress().tolist() ,
            "stds": [ [0.,0.,0.] for _ in range(len(struct))], "local_energy_stds" : [ 0. for _ in range(len(struct))] , "stress_stds" : [0. for _ in range(6)]})
        })
    return structdict

def dict2ase(structdict : dict) -> Atoms :
    """
    Takes the FLARE dictionary and returns the ASE atoms object
    """
    struct = Atoms( numbers = structdict["numbers"],
                   positions = structdict["positions"],
                   cell = structdict["cell"],
                   pbc = structdict["pbc"])
    forces = structdict["results"]["forces"]
    energy = structdict["results"]["energy"][0]
    calc = SinglePointCalculator(atoms=struct, energy=energy, forces=forces, stress=[ 0 for _ in range(6) ] )
    struct.calc = calc

    return struct


def huber_loss(hyperparameters,gp_model,weights):
    """
    Compute huber loss of the model on its training set given some hyperparameters
    Losses on energies,forces and stresses are then multiplied by weights
    """

    # Residuals on energies,forces and stresses
    # Initialize empty arrays

    delta_e = 0.05
    delta_f = 0.05
    delta_s = 0.001

    omega_e,omega_f , omega_s = weights

    e_residuals = np.empty(0)
    f_residuals = np.empty(0)
    s_residuals = np.empty(0)
    gp_model.set_hyperparameters(hyperparameters)
    gp_model.update_matrices_QR()

    n_forces = 0
    n_structs = len(gp_model.training_structures)

    for train_struct in gp_model.training_structures:
        nat = len(train_struct.species)
        n_forces += nat*3
        gp_model.predict_local_uncertainties(train_struct)
        prediction = train_struct.mean_efs
        e_residuals = np.hstack((e_residuals,(train_struct.energy - prediction[0])/nat))
        f_residuals = np.hstack((f_residuals,(train_struct.forces - prediction[1:-6])))
        #s_residuals = np.hstack((s_residuals,(train_struct.stress - prediction[-6:])))

    e_loss = huber(delta_e, e_residuals).sum()
    f_loss = huber(delta_f, f_residuals).sum()
    #s_loss = huber(delta_s, s_residuals)


    loss = omega_e * e_loss / n_structs + omega_f * f_loss / n_forces # + omega_s * s_loss / n_structs*6
    hyps_lasso = 1.
    omega_h = 1e-7
    for hyp in hyperparameters[1:3]:
        print(hyp)
        hyps_lasso *= 1/abs(hyp)
    print(hyps_lasso)
    loss += omega_h * hyps_lasso
    print(loss)

    return loss


def get_dft_data(conf):
    """"
    get data in the right format for flare training from an ase atoms object
    """

    #get DFT data in the proper format
    dft_energy = conf.get_potential_energy()
    dft_forces = conf.get_forces()
    dft_stress = conf.get_stress()

    # Convert ASE stress (xx, yy, zz, yz, xz, xy) to FLARE stress
    # (xx, xy, xz, yy, yz, zz).
    flare_stress = None
    if dft_stress is not None:
        flare_stress = -np.array(
            [
                dft_stress[0],
                dft_stress[5],
                dft_stress[4],
                dft_stress[1],
                dft_stress[3],
                dft_stress[2],
            ]
        )

    return dft_energy, dft_forces, flare_stress


def train_offline(config, train_set, test_set):

    #initialize variables and objects
    #add write to json
    #make better use of logging, export data such as which sparse env you are selecting

    #use function from flare package
    #set stress_training in the config file, eventually
    flare_calc, kernels = get_sgp_calc(config["flare_calc"])
    #returns a flare ASE calculator and the kernels

    optimize_every = config["optimize_every"]
    min_optimize = config["min_optimize"]
    max_optimize = config["max_optimize"]

    call_threshold = config["call_threshold"]
    add_threshold  = config["add_threshold"]

    nr_initial_envs = config["nr_initial_envs"]

    species_code = config["species_code"]
    isolated_energies = config["isolated_energies"]

    files_prefix = config["files_prefix"]

    sparse_indices      = []
    training_structures = []

    oracle_calls = 1
    nsparse      = 0
    last_optim   = 0
    step         = 0

    file_log    = open(f"training_{files_prefix}.log",'w')
    file_hyps   = open(f"hyps_{files_prefix}.dat",'w')
    file_lik    = open(f"lik_{files_prefix}.dat",'w')
    file_maes_e = open(f"e_maes_{files_prefix}.dat",'w')
    file_maes_f = open(f"f_maes_{files_prefix}.dat",'w')

    #log initial stuff
    file_log.write("You are running Offline Learner version 4-11-2025\n")
    file_log.write("Author : Davide Alimonti + Gilberto Nardi, nanoMLMS @ University of Milan\n")

    dateformat = "%d/%m/%Y %H:%M:%S"
    now = datetime.now()

    file_log.write(f"Execution started at {now.strftime(dateformat)}\n")
    file_log.write(" * * * * * * * \n")

    file_log.write("DATASET DETAILS\n")
    file_log.write(f'train set has {len(train_set)} files.\n')
    file_log.write(f'test set has {len(test_set)} files.\n')

    print(f"Running with files_prefix {files_prefix}")
    file_maes_e.write("# Begin here\n")
    file_maes_f.write("# Begin here\n")

    #TODO: for backwards compatibility
    #test_set_flare_struc = [ ase2flare() for a in test_set]



    #start training
    print("Training...")
    file_log.write('Training...')

    ######
    #main loop
    ######

    #for conf in tqdm(train_set):
    for i, conf in enumerate(train_set):
        file_log.write(f"   - Frame nr {step} \n")

        flare_conf      = FLARE_Atoms.from_ase_atoms(conf, copy_calc_results=True)#FLARE_Atoms.from_ase_atoms(conf)#ase2flare(struct, species_code, isolated_energies) #or FLARE_Atoms.from_ase_atoms()?
        flare_conf.calc = flare_calc

        energy, forces, stress = get_dft_data(conf)
        sgp = flare_calc.gp_model#.sparse_gp

        #initial step
        if step==0:
        
            #choose at random the indices for sgp initialization
            indices = np.random.choice(len(conf), nr_initial_envs, replace=False)
            sgp.update_db(flare_conf, forces=forces, energy=energy, stress=stress, custom_range=indices)
    
            #log / store info
            nsparse += nr_initial_envs
            sparse_indices.append(indices.tolist())
            conf.info["sparse_set"] = np.array(indices)
            training_structures.append(conf)
            file_log.write('initialized gp with radnom enviornments: atoms' + np.array2string(indices) + '\n')
            file_log.flush()

            #end first step
            step+=1
            continue

        else:

            #compute uncertainties
            flare_conf.calc.calculate(atoms=conf, properties='stds')
            stds = flare_conf.calc.results.get("stds", np.zeros_like(forces))
            
            if np.max(stds)>call_threshold:
                oracle_calls +=1

                #get high uncertainty configs
                indices = np.where(stds>add_threshold)[0]
                sgp.update_db(flare_conf, forces=forces, energy=energy, stress=stress, custom_range=indices) #differnt from davide, but as in flare-otf. Could it be differnt for the full set?

                file_log.write("Added environments: \n" + np.array2string(indices) + '\nUncertainties: \n' + np.array2string(stds[indices]) +'\n')
                sparse_indices.append(indices.tolist())
                nsparse += len(indices)
                conf.info["sparse_set"] = np.array(indices)
                training_structures.append(conf)

                if oracle_calls > min_optimize and oracle_calls < max_optimize and oracle_calls%optimize_every == 0:

                    file_log.write(f"optimizing call #{oracle_calls}...")
                    #train hyperparameters
                    rollback = optimize_hyps(sgp.sparse_gp, **config["optimizer_options"])

                    if rollback: #optimization failed
                        file_log.write('optimization failed. Currently only accept style rollback is implemented.\n')
                        if config["when_rollback"] == "discard":
                            #create an sgp with all configs up until this one
                            print('discard style rollback is not yet implemented')
                            pass
                        pass
                    else:
                        file_hyps.write(f"{step}\t{' '.join(map(str, sgp.sparse_gp.hyperparameters))}\n")
                        file_hyps.flush()
                        #all ok

                        #del training_structures[-1]
                else:
                    pass

                neglik,_= compute_negative_likelihood_grad_stable(sgp.sparse_gp.hyperparameters, sgp.sparse_gp, precomputed=False)

                #log_errors(sgp.sparse_gp, test_set) TODO - for backwards compatibility
                #check errors for the learning curve
                errors = compute_test_errors(flare_calc, 'test.xyz', config["isolated_energies"][str(test_set[0].numbers[0])], use_norm=False, tqdm_extra_string=f'step={step}')
                log_errors_gibo(errors, file_maes_e, file_maes_f, step)

                file_lik.write(f"{step}\t{neglik}\t{nsparse}\n")
                file_lik.flush()

                file_log.flush()


            step += 1

    ########
    #end of main loop
    ########

    #build map and write model - only here?
    flare_calc.build_map()
    flare_calc.write_model(config['files_prefix']+'_model.json') #this way is missing some stuff? like results? could have to be fixed

    write('added_structures.xyz', training_structures)

    #end logging
    now = datetime.now()
    file_log.write(" * * * * * * * \n")
    file_log.write(f"Execution ended at {now.strftime(dateformat)}\n")

    #close logging files
    file_log.close()
    file_hyps.close()
    file_lik.close()
    file_maes_e.close()
    file_maes_f.close()

    return

def model_from_dict(json_dict_file):
    #from flare.learners.OTF class
    #flare_calc_dict = json.load(open(json_dict_file))["flare_calc"]

    # Build FLARE_Calculator from dict
    #if flare_calc_dict["class"] == "FLARE_Calculator":
    #     flare_calc = FLARE_Calculator.from_file(json_dict_file)
    #     _kernels = None
    #     # Build SGP_Calculator from dict
    #     # TODO: we still have the issue that the c++ kernel needs to be
    #     # in the current space, otherwise there is Seg Fault
    #     # That's why there is the _kernels
    # elif flare_calc_dict["class"] == "SGP_Calculator":
    #     flare_calc, _kernels = SGP_Calculator.from_file(json_dict_file)
    # else:
    #     raise TypeError(f"The calculator {json_dict_file} is not recognized.")


    #todo: generalize

    flare_calc, _kernels = SGP_Calculator.from_file(json_dict_file)
    return flare_calc, _kernels

def main(config_file):

    config = yaml.safe_load(open(config_file, 'r'))

    train, test = make_sets(config)
    train_offline(config, train, test)

if __name__=='__main__':

    main(sys.argv[1])