#create a dataset in different ways and train a potential with flare
#using ase to deal with data loading
#add extra files (e.g. detached atoms) and a final main function to produce a train and test set

import numpy as np
from ase.io import read, write
import random
import sys
import yaml
from collections import Counter

def make_random_sets(files, test_ratio=0.1, valid_ratio=0.0, seed=None, per_file_splitting=True, verbose=False):
    #per file splitting: split in test, train and validation for each file rather than at random from the full set of all provided configurations

    if seed is not None:
        random.seed(seed)
    
    #compatibilitiy with single file
    if isinstance(files, str):
        files = [files]

    train_set = []
    test_set  = []
    valid_set = []

    all_configs = []

    for f in files:
        configs = read(f, index=':')
 
        if per_file_splitting:
 
            random.shuffle(configs)

            valid_index = int(len(configs)*valid_ratio)
            test_index  = int(len(configs)*test_ratio) + valid_index
            
            #check this
            if valid_index ==1:
                valid_set += [configs[0]]
            elif valid_index > 1:
                valid_set += configs[:valid_index]
    
            if test_index ==valid_index+1:
                test_set += [configs[valid_index]]
            elif test_index > valid_index+1:
                test_set += configs[valid_index:test_index]
            
            train_set += configs[test_index:]

            if verbose:
                print(f'file {f}: {len(configs)-test_index} configs in train set, {test_index-valid_index} configs in test set, {valid_index} configs in valid set.')

        else:
            all_configs += configs
    

    if not per_file_splitting:
        random.shuffle(all_configs)

        valid_index = int(len(all_configs)*valid_ratio)
        test_index  = int(len(all_configs)*test_ratio) + valid_index        

        valid_set = all_configs[:valid_index]
        test_set  = all_configs[valid_index:test_index]
        train_set = all_configs[test_index:]

        if verbose:
            print(f' all files together: {len(train_set)} configs in train set, {len(test_set)} configs in test set, {len(valid_set)} configs in valid set.')
    

    return train_set, test_set, valid_set


def make_sequential_sets(bulk_files, surf_files, cluster_files, test_ratio=0.1, valid_ratio=0.0, seed=None, per_file_splitting=True):

    train_bulk, test_bulk, valid_bulk = make_random_sets(bulk_files, test_ratio, valid_ratio, seed, per_file_splitting)
    train_surf, test_surf, valid_surf = make_random_sets(surf_files, test_ratio, valid_ratio, seed, per_file_splitting)
    train_cluster, test_cluster, valid_cluster = make_random_sets(cluster_files, test_ratio, valid_ratio, seed, per_file_splitting)

    return [train_bulk, train_surf, train_cluster], [test_bulk, test_surf, test_cluster], [valid_bulk, valid_surf, valid_cluster]
    


def make_sparsely_injected_sets(files, files_for_injection, injection_frequency, stop_inject, test_ratio=0.1, seed=None, shuffle_injections=False):
    #shuffle injections if you want to 
    #no validation set as the order of configs only matters when using gpr

    train_set, test_set = [], []

    for f in files:

        trs, tes, _ = make_random_sets(f, test_ratio=test_ratio, seed=seed)
        train_set += trs
        test_set += tes

    all_injections = []
    for f in files_for_injection:
        frames = read(f, index=':')
        all_injections.append(frames)
    assert len(all_injections)==len(files_for_injection)

    #all_injections is a list of lists. The structure should be modified a bit if you want to really 
    #shuffle all configs.
    #if shuffle_injections:
    #    random.shuffle(all_injections)

    final_train_set = []
    injections_iter = 0

    for i, conf in enumerate(train_set):

        final_train_set.append(conf)

        if i % injection_frequency == 0 and i<stop_inject:

            # Keep trying until we find a non-empty injection list or all are exhausted
            while all_injections:

                inj_list = all_injections[injections_iter % len(all_injections)]

                if inj_list:  # non-empty
                    inj_conf = inj_list.pop()
                    final_train_set.append(inj_conf)
                    injections_iter += 1
                    break
                else:  # empty, remove it from the master list
                    all_injections.pop(injections_iter % len(all_injections))

            # if all_injections is empty, nothing is injected and we just continue
    
    print(f'final train set: {len(final_train_set)} configs of which {injections_iter} have been injected.')

    remaining_injections = []

    for inj_list in all_injections: 
        remaining_injections += inj_list  # add all remaining configs

    return final_train_set, test_set, remaining_injections

###What's a good measure of disorder in atomistic configurations?
#some experimental utilities: sort configurations by how ordered they are (type 1: GCN stddev, type 2: CNAPs number)

def sort_by_gcn(conf_set, cutoff, pbc=True):
    """
    sort configs in set by increasing spread of the generalized coordination number of the atoms in the frames,
    quantified as the standard deviation of the gcns
    """
    from snow.descriptors.coordination import agcn_calculator

    gcn_sds = np.zeros(len(conf_set))

    for i, frame in enumerate(conf_set):
        _, gcns = agcn_calculator(frame.get_positions(), cutoff, pbc=pbc, box=frame.get_cell())
        sd = np.std(gcns)
        gcn_sds[i] = sd
    
    ids = np.argsort(gcn_sds)
    gcn_sds = gcn_sds[ids]
    sorted_frames = [conf_set[i] for i in ids]

    return sorted_frames, gcn_sds, ids

def sort_by_cnap_number(frames, cutoff, pbc=True):
    """
    sort by the number of distinct cnaps normalized by the number of atoms in the system.
    """
    
    from snow.descriptors.cna import cna_peratom

    cnap_ratios = np.zeros(len(frames))
    
    for i, frame in enumerate(frames):  
        cnas = cna_peratom(frame.get_positions(), cutoff, pbc, box=frame.get_cell())
        pattern_counter = Counter()

        #expand signatures by their count and sort to get permutation invariance
        for signs, counts in cnas:
            expanded = []
            for sig, cnt in zip(signs, counts):
                expanded.extend([tuple(sig)] * cnt) #extend appends the tuples as single elements and avoids the creation of a nested list
            
            pattern = tuple(sorted(expanded)) #sort for perm invariance
            pattern_counter[pattern] += 1

        cnap_ratios[i] = len(pattern_counter) / len(frame)

    ids    = np.argsort(cnap_ratios)
    sorted_frames = [frames[i] for i in ids]  

    return sorted_frames, cnap_ratios[ids], ids

def split_bulk_surf_clust(atoms):
    """attempt separating different structures (bulk, surfaces, clusters) based
    on the number density of atoms (generally three ranges are present)
    
    notice: this generally works if you have performed DFT calculations with plane waves, 
    so you have to include vacuum and the amount of total vaucum depends on the configuration type
    
    This sometimes does not work. check it out
    """

    bulk, surf, clust = [], [], []
    densities = []
    for a in atoms:
        Vcell = a.get_cell().volume
        N = float(len(a))
        densities.append(N / Vcell)

    # Find two thresholds using kmeans-style split on sorted densities
    sorted_dens = sorted(set(densities))
    
    #llm stuff
    def find_split(values):
        """Find the best split point that maximizes inter-group distance."""
        best_split = None
        best_gap = -1
        for i in range(len(values) - 1):
            gap = values[i + 1] - values[i]
            if gap > best_gap:
                best_gap = gap
                best_split = (values[i] + values[i + 1]) / 2.0
        return best_split

    # Find two split points dividing densities into three groups
    if len(sorted_dens) < 2:
        return atoms, [], []

    split1 = find_split(sorted_dens)
    if split1 is None:
        return atoms, [], []

    lower = [v for v in sorted_dens if v < split1]
    upper = [v for v in sorted_dens if v >= split1]
    split2 = find_split(upper) if len(upper) > 1 else None

    for a, d in zip(atoms, densities):
        if split2 is not None:
            if d < split1:
                clust.append(a)
            elif d < split2:
                surf.append(a)
            else:
                bulk.append(a)
        else:
            if d < split1:
                clust.append(a)
            else:
                bulk.append(a)

    return bulk, surf, clust

def make_sets(config):
    """
    use info in the config_yaml object to create a train and test set using the functions defined above.
    
    implemented dataset styles:
    ordered (shuffle per-class, keep class order fixed - e.g. bulk, then surf, then clusters)
    random (all shuffled) --- DEFAULT
    injected (like ordered, with some files injected periodically - eveery injection_frequency - until the stop_inject config.)
    intact: leave the order given in the single files
    cnap: sort by increasing number of cnaps over number of atoms
    gcn_spread: sort by standard deviation of gcn in the configurations. 
    """

    style = config.get("dataset_style", 'random') #default style is random
    seed  = config.get("seed")

    if style == "injected":

        injection_frequency = config.get('injection_frequency', 4)
        stop_inject         = config.get('stop_inject', 60)

        files = config.get('bulk_files') + config.get('surf_files') + config.get('clusters_files')
        train, test, _ = make_sparsely_injected_sets(files, 
                                                    config.get("injection_files"),
                                                    injection_frequency=injection_frequency,
                                                    stop_inject= stop_inject,
                                                    seed=config.get("seed"))

    elif style == "random":

        files = config.get("bulk_files",[]) + config.get("surf_files",[]) + config.get("clusters_files",[]) + config.get("extra_files",[])
        train, test, _ = make_random_sets(files, seed=seed)

    elif style == "ordered":

        bulks  = config.get('bulk_files')
        surfs  = config.get('surf_files')
        clusts = config.get('clusters_files')
        [train_b, train_s, train_c], [test_b, test_s, test_c], _ = make_sequential_sets(bulks, surfs, clusts, seed=seed)

        train = train_b + train_s + train_c
        test  = test_b + test_s + test_c
    
    elif style == 'intact':
        train = read(config.get('train_file'), index=':')
        test  = read(config.get('test_file'), index=':')

    elif style == 'cnap':

        #just like random
        files = config.get("bulk_files",[]) + config.get("surf_files",[]) + config.get("clusters_files",[]) + config.get("extra_files",[])
        train, test, _ = make_random_sets(files, seed=seed)

        #sort by cnap number over atoms number
        cutoff = config.get('solvation_shell_cutoff')
        train, _, _  = sort_by_cnap_number(train, cutoff)
        test, _, _   = sort_by_cnap_number(test, cutoff)

    elif style == 'gcn':

        #just like random
        files = config.get("bulk_files",[]) + config.get("surf_files",[]) + config.get("clusters_files",[]) + config.get("extra_files",[])
        train, test, _ = make_random_sets(files, seed=seed)

        #sort by gcn distribution standard deviation
        cutoff = config.get('solvation_shell_cutoff')   
        train, _, _  = sort_by_gcn(train, cutoff)
        test, _, _   = sort_by_gcn(test, cutoff)

    else:
        sys.exit(f'style {style} is not implemented.')


    #if style is intact you already have your train set and test set. otherwise, export them.
    if not style == 'intact':
        write('train.xyz', train)
        write('test.xyz', test)
    
    return train, test


if __name__ == '__main__':

    with open(sys.argv[1], 'r') as f:
        config = yaml.safe_load(f)
    make_sets(config)
