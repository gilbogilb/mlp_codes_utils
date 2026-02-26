#create a dataset in different ways and train a potential with flare
#using ase to deal with data loading

import numpy as np
from ase.io import read, write
import random
import sys
import yaml

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




    






if __name__ == '__main__':

    folder = '/Users/ginardi/Desktop/science/dft/DFT_xyz_45ry/'
    files  = [folder+'bulk.xyz', folder+'surf.xyz', folder+'clusters.xyz']
    injections = [folder+'cu-relax-100.xyz', folder+'isomers_relaxations.xyz']

    tr, te, _ = make_sparsely_injected_sets(files, injections, 4, 60)
    write('train.xyz', tr)