import numpy as np
import sys
import time
from flare.bffs.sgp.calculator import SGP_Calculator
from flare.bffs.sgp._C_flare import   B2, NormalizedDotProduct, SparseGP, Structure
from flare.bffs.sgp import SGP_Wrapper
from flare.learners.otf import OTF
from flare.io import otf_parser
from flare.scripts.otf_train import get_gp_calc
import json
import tempfile
import copy
from ase.io import read,write
from scipy.optimize import minimize
import flare
import random
import sys
import yaml
import copy
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from datetime import datetime
from scipy.special import huber

try:
    from tqdm import tqdm
except:
    def tqdm(iterable):
        return iterable

def ase2flare(struct,species_code,isolated_energies):
    """
    Takes an ASE structure and returns a FLARE structure object
    """
    noa = len(struct.numbers)
    coded_species=[]
    eisol = 0
    for spec in struct.numbers:
        coded_species.append(species_code[str(spec)])
        eisol += isolated_energies[str(spec)]
    cell = struct.cell.array
    pos = struct.positions
    flare_struct = Structure(cell,list(coded_species),pos, cutoff, descriptors)
    flare_struct.wrap_positions()
    if "forces" in struct.arrays:
        flare_struct.forces = struct.arrays["forces"].reshape(-1)
    else:
        flare_struct.forces = struct.get_forces().reshape(-1)
    flare_struct.energy = np.array([struct.calc.get_potential_energy() - eisol])
    return flare_struct

def check_mae(gp_model,train_struct):
    """
    Returns the error on energy/per atom and the array of forces errors incurred by the gp_model on a FLARE structure train_struct
    """
    force_components_errors =[]
    nat=len(train_struct.species)
    gp_model.predict_local_uncertainties(train_struct)
    energy_error = (train_struct.energy -  train_struct.mean_efs[0]) / nat
    force_components_errors = (train_struct.forces - train_struct.mean_efs[1:-6]).tolist()
    return energy_error, np.array(force_components_errors)

def log_errors(gp_model,testsets):
    """
    Measures the error incurred by the potential on a collection of testsets (made of FLARE structures). 
    Update: also writes extra-data (nsparse, n_dft_calls) to a different file
    """
    file_maes_e.write(f"{step}\t")
    file_maes_f.write(f"{step}\t")
    for testset in testsets:
        enerrs,fcerrs = np.empty(0),np.empty(0)
        for test_struct in testset:
            enerr, fcerr= check_mae(gp_model,test_struct)
            enerrs  = np.concatenate((enerrs,enerr))
            fcerrs  = np.concatenate((fcerrs,fcerr))
        mae_e = np.mean(np.abs(enerrs))
        mae_f = np.mean(np.abs(fcerrs))
        file_maes_e.write(f"{mae_e:.5f}\t")
        file_maes_f.write(f"{mae_f:.5f}\t")
    file_maes_e.write("\n")
    file_maes_f.write("\n")
    file_maes_e.flush()
    file_maes_f.flush()
    return

def compute_negative_likelihood_grad_stable(
    hyperparameters, sparse_gp, precomputed=False
):
    """
    Compute the negative log likelihood and gradient with respect to the
    hyperparameters.
    """

    assert len(hyperparameters) == len(sparse_gp.hyperparameters)

    sparse_gp.set_hyperparameters(hyperparameters)

    negative_likelihood = -sparse_gp.compute_likelihood_gradient_stable(precomputed)
    negative_likelihood_gradient = -sparse_gp.likelihood_gradient

    return negative_likelihood, negative_likelihood_gradient

def optimize_hyps(gp_model,opt_method,minhyps,maxhyps,max_iterations,bounds,gtol,loss_function_config):
    """
    Finds optimal hyperparameters for the gp_model, using its current hyps as starting guess.
    If the found hyps are outside the defined minhyps,maxhyps, it will return True - signalling a failure
    otherwise it will set new hyps to the model
    """
    rollback = False
    initial_guess = gp_model.hyperparameters
    old_hyps      = np.array(initial_guess) # for rollback
    if loss_function_config["name"] == "negative_likelihood":
        loss_function = compute_negative_likelihood_grad_stable
        arguments = (gp_model,True)
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
                    "gtol":gtol ,
                    "maxiter": max_iterations,
                    "eps" : np.array([1e-3,1e-4,1e-4,1e-5])
                }
            )
    print(optimization_result)
    # Assign likelihood gradient, if it didn't explode
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
        file_log.write("Optimization resulted in exploded or collapsed hyps!\n")
        file_log.write(f"Would have been : {np.array2string(np.abs(optimization_result.x))}"+'\n')
        rollback = True
        file_log.write("Hyps NOT updated\n")
    return rollback

def write_to_json(gp_model,power,radial_basis_type,
                  cutoff_function,cutoff,nspecies,nmax,lmax,
                  opt_method,variance_type,max_iterations,
                  minhyps, maxhyps,bounds,
                  sigma_e,sigma_f,sigma_s,sigma,
                  isolated_energies_mapped,descriptor_type,gtol,loss_function_config):
    """
    Returns a JSON of the model including all necessary data to retrain it.
    It also produces the maps of the model
    """
    hyperlist=np.array(gp_model.hyperparameters).tolist()
    #log_errors(gp_model,testsets)
    gp_model.write_mapping_coefficients(f"{files_prefix}_coeffs.dat","davide",0)
    gp_model.write_sparse_descriptors(f"{files_prefix}_sparse_desc.dat","davide")
    gp_model.write_L_inverse(f"{files_prefix}_inv.dat","davide")
    gpmodeldict = dict({"sparse_indice": [sparse_indices], "training_structures": training_structures})
    gpmodeldict["cutoff"] = cutoff
    gpmodeldict["species_map"] = species_code
    gpmodeldict["variance_type"] = variance_type
    gpmodeldict["single_atom_energies"] = isolated_energies_mapped
    gpmodeldict["energy_training"] = True
    gpmodeldict["force_training"] = True
    gpmodeldict["stress_training"] = False
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
        sigma,power,radial_basis_type,cutoff_function,cutoff,
        nspecies,nmax,lmax,
        sigma_e,sigma_f,sigma_s) :
    """
    Create an empty model with working kernels.
    Also creates the kernel and descriptor objects.
    """
    kernels = [ NormalizedDotProduct(sigma , power) ]
    descriptors = [ B2(radial_basis_type,cutoff_function,[0,cutoff],[] , [nspecies , nmax, lmax]) ]
    gp_model_init = SparseGP( kernels, sigma_e , sigma_f, sigma_s)
    return gp_model_init,descriptors ,kernels

def configurations_order(configurations_list : list ,random_shuffle : bool, shuffle_seed : int ) -> list:
    """
    If random_shuffle is false, it will return configurations_list as provided.
    If random_shuffle is true , it will return the list shuffled with seed shuffle_seed
    """
    if random_shuffle:
        random.seed(shuffle_seed)
        random.shuffle(configurations_list)
    return configurations_list

def model_from_dict(structuresdict,sparse_indices,species_code,hyps,modelstruct):
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
        "results"  : dict({"forces": struct.arrays["forces"].tolist() if "forces" in struct.arrays else struct.get_forces().tolist() , "energy" : [struct.calc.get_potential_energy().tolist()], "stress": struct.get_stress().tolist() ,
            "stds": [ [0.,0.,0.] for _ in range(noa)], "local_energy_stds" : [ 0. for _ in range(noa)] , "stress_stds" : [0. for _ in range(6)]})
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

config_file = sys.argv[1]
config = yaml.safe_load(open(config_file,'r'))

np.random.seed(config["seed"])
random.seed(config["seed"])

gp_model,descriptors,kernels = initialize_gp(**config["gp_config"])

species_code = config["species_code"]
isolated_energies = config["isolated_energies"]
cutoff = config["gp_config"]["cutoff"]
optimize_every = config["optimize_every"]

trainset =[]
testsets = []

files_prefix = config["files_prefix"]
file_log  =open(f"training_{files_prefix}.log",'w')
file_hyps =open(f"hyps_{files_prefix}.dat",'w')
file_lik  =open(f"lik_{files_prefix}.dat",'w')
#file_uncs =open(f"lik_{files_prefix}.dat",'w')

# PRINT VERSION
file_log.write("You are running Offline Learner version 4-11-2025\n")
file_log.write("Author : Davide Alimonti , nanoMLMS @ University of Milan\n")

dateformat = "%d/%m/%Y %H:%M:%S"
now = datetime.now()

file_log.write(f"Execution started at {now.strftime(dateformat)}\n")
file_log.write(" * * * * * * * \n")

file_log.write("DATASET DETAILS\n")

ntrain_all = 0
overall_testset= []

if config["random_shuffle"]:
    print("Configurations are randomly shuffled")
    file_log.write("Configurations are randomly shuffled\n")

#create test and train set
for dataset in config["datasets"]:
    name=dataset["name"]
    if not config["random_shuffle"]:
        print(f"Pointer {ntrain_all} , begins {name}")
        file_log.write(f"Pointer {ntrain_all} , begins {name}\n")
    configurations = []
    ntest = dataset["ntest"]
    ntrain= dataset["ntrain"]
    for f in dataset["files"]:
        for atoms in read(f,index=":"):
            configurations.append(atoms)
    random.seed(config["seed"])
    random.shuffle(configurations)

    if ntest:
        flareset=[]
        for struct in configurations[:ntest] :
            flareset.append(ase2flare(struct,species_code,isolated_energies))
            write("testset.xyz",struct,append=True)
        testsets.append(flareset)
        overall_testset += flareset

    if ntrain > 0 :
        trainset += configurations[ntest:ntest+ntrain+1]
        ntrain_all += ntrain
    elif ntrain == -1 : #if ntrain is -1, add all configurations except those in test
        trainset += configurations[ntest:]
        ntrain_all += len(configurations[ntest:])

print(f"Total train: {ntrain_all}")
testsets = [overall_testset] + testsets

dataset_names = [ dataset["name"] for dataset in config["datasets"]]
dataset_names = ['average'] + dataset_names

#compute mav
if overall_testset:
    for idts,ts in enumerate(testsets):
        frc_mav = 0.0
        ene_mav = 0.0
        nats    = 0
        nfrcs   = 0
        for struct in ts:
            frc_mav += np.sum(np.abs(struct.forces))
            ene_mav += np.abs(struct.energy)
            nats    += len(struct.species)
            nfrcs   += len(struct.species)*3
        ene_mav = ene_mav/nats
        frc_mav = frc_mav/nfrcs
        print(f"{dataset_names[idts]} ({idts}), frc mav {frc_mav} , ene mav {ene_mav}")

print(f"Running with files_prefix {files_prefix}")
file_maes_e=open(f"e_maes_{files_prefix}.dat",'w')
file_maes_f=open(f"f_maes_{files_prefix}.dat",'w')
file_maes_e.write("# Begin here\n")
file_maes_f.write("# Begin here\n")


step = 0
print("Training")

skip_frac=0
min_optimize = config["min_optimize"]
max_optimize = config["max_optimize"]


sparse_indices=[]
training_structures=[]

call_threshold = config["call_threshold"]
add_threshold  = config["add_threshold"]


trainset = configurations_order(trainset, config["random_shuffle"], config["shuffle_seed"])
oracle_calls=0
nsparse = 0
last_optim = 0

write('trainset.xyz',trainset)
file_log.write(" * * * * * * * \n")
file_log.write("TRAINING STARTS\n")
for struct in tqdm(trainset) :
    file_log.write(f"   - Frame nr {step} \n")
    if step and np.random.rand() < skip_frac :
        file_log.write(f"Skipped step {step}\n")
        step+=1
        continue
    poten= struct.calc.get_potential_energy()/ len(struct.numbers)
    maxfrc = max( struct.arrays["forces"] if "forces" in struct.arrays else struct.get_forces(), key = lambda x : np.linalg.norm(x))
    noa = len(struct.numbers)
    flare_struct = ase2flare(struct,species_code,isolated_energies)
    flare_struct.compute_descriptors()

    #first step
    if not step:
        nr_initial_envs = config["nr_initial_envs"]
        gp_model.add_training_structure(flare_struct)
        indices = np.random.choice(len(struct),nr_initial_envs,replace=False)
        nsparse += nr_initial_envs
        gp_model.add_specific_environments(flare_struct,indices)
        gp_model.update_matrices_QR()
        #log_errors(gp_model,testsets)
        sparse_indices.append(indices.tolist())
        structdict=ase2dict(struct)
        training_structures.append(structdict)
        file_log.write("First step taken\n")
        file_log.write("Added environments: \n")
        file_log.write(np.array2string(indices))
        file_log.write('\n')
        file_log.flush()
        gp_model.precompute_KnK()
        neglik,_= compute_negative_likelihood_grad_stable(gp_model.hyperparameters, gp_model, precomputed=False)
        log_errors(gp_model,testsets)
        file_lik.write(f"{step}\t{neglik}\t{nsparse}\n")
        file_hyps.write(f"{step}\t{' '.join(map(str, gp_model.hyperparameters))}\t{neglik}\n")
        step+=1
        continue

    # If step > 0, run normally
    gp_model.predict_local_uncertainties(flare_struct)
    sigma = gp_model.hyperparameters[0]
    uncs= np.array(flare_struct.local_uncertainties)[0]
    uncs= np.sqrt(np.abs(uncs))/np.abs(sigma) # Rooted, unitless "std dev", consistent with flare-otf. Take abs because for numerical instabilities, some can be negative
    #for u in uncs:
    #    file_uncs.write(str(u)+' ')
    #file_uncs.write('\n')
    if np.max(uncs) > call_threshold :
        oracle_calls += 1
        indices = np.where(uncs > add_threshold)[0]
        if len(indices):
            gp_model.add_training_structure(flare_struct)
            gp_model.add_specific_environments(flare_struct, indices)
            file_log.write("Avg. en, largest force:\n")
            file_log.write(f"{poten}\t{np.array2string(maxfrc)}\n")
            file_log.write("Added environments: \n")
            file_log.write(np.array2string(indices))
            file_log.write('\n')
            file_log.write("Uncertainties: \n")
            file_log.write(np.array2string(uncs[indices]))
            file_log.write('\n')
            file_log.flush()
            gp_model.update_matrices_QR()
            sparse_indices.append(indices.tolist())
            structdict=ase2dict(struct)
            training_structures.append(structdict)
            nsparse += len(indices)
            if oracle_calls < max_optimize and oracle_calls > min_optimize and (oracle_calls-min_optimize)%optimize_every == 0 :
                rollback = optimize_hyps(gp_model,**config["optimizer_options"])
                file_log.write(f"Attempting optimization for oracle call nr. {oracle_calls}")
                file_log.write('\n')
                file_log.flush()
                if rollback: #Optimization failed
                    if config["when_rollback"] == "discard" : #Rollback hyps AND trainset
                        oracle_calls -= 1
                        nsparse -= len(indices)
                        del training_structures[-1]
                        del sparse_indices[-1]
                        modelstruct = [config["gp_config"]["power"],
                            config["gp_config"]["nspecies"],
                            config["gp_config"]["nmax"],
                            config["gp_config"]["lmax"],
                            config["gp_config"]["cutoff"]]
                        gp_model,descriptors,kernels = model_from_dict(training_structures, sparse_indices, species_code,gp_model.hyperparameters,modelstruct)
                    # Hyps are rolled back automatically
                    # So here we just communicate that we failed
                    file_log.write("Model NOT updated. New (old) hyps : \n")
                    file_log.write('\n')
                    file_log.write(str(gp_model.hyperparameters)+'\n')
                    file_log.write("Mode was "+config["when_rollback"]+'\n')
                else: # Model updated correctly
                    file_log.write("Model updated. New hyps : \n")
                    file_log.write('\n')
                    file_log.write(str(gp_model.hyperparameters)+'\n')
                    file_log.flush()
                    last_optim = step
                    neglik,_= compute_negative_likelihood_grad_stable(gp_model.hyperparameters, gp_model, precomputed=False)
                    file_hyps.write(f"{step}\t{' '.join(map(str, gp_model.hyperparameters))}\t{neglik}\n")
                file_hyps.flush()
            gp_model.update_matrices_QR()
            gp_model.precompute_KnK()
            neglik,_= compute_negative_likelihood_grad_stable(gp_model.hyperparameters, gp_model, precomputed=False)
            log_errors(gp_model,testsets)
            file_lik.write(f"{step}\t{neglik}\t{nsparse}\n")
            ntrstructs = len(gp_model.training_structures)
            file_log.write(f"Nr. of training structures is now: {ntrstructs}\n")
            file_lik.flush()
            file_log.flush()
    gp_model.update_matrices_QR()
    step+=1

    if step%10 == 0:
        print('read',step,'configurations...')

now = datetime.now()
file_log.write(" * * * * * * * \n")
file_log.write(f"Execution ended at {now.strftime(dateformat)}\n")

write_to_json(gp_model,**config["gp_config"],**config["optimizer_options"],variance_type="local", isolated_energies_mapped= config["isolated_energies_mapped"],descriptor_type="B2")

fout = open('added_structures.xyz','w')
fout.write('')
fout.close()

for structdict,spids in zip(training_structures,sparse_indices):
    #Write the structure that were added to the effective set to a file
    #It will include metadata indicating which atoms were added to sparse set
    struct= dict2ase(structdict)
    struct.info["sparse_set"] = np.array(spids)
    write("added_structures.xyz",struct,append=True) #file should be flushed first

