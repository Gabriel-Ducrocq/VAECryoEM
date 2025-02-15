import os
import mrc
import sys
path = os.path.abspath("model")
sys.path.append(path)
import yaml
import torch
import utils
import pickle
import argparse
from ctf import CTF
import numpy as np
from tqdm import tqdm
from polymer import Polymer

parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--path', type=str, required=True)
parser_arg.add_argument('--chain_information', action=argparse.BooleanOptionalAction)
args = parser_arg.parse_args()
path = args.path
chain_information = args.chain_information


def compute_distance(polymer, idxA, idxB, chain_id):
    """ Computes the distance between the two upper domains"""
    domainA = polymer.coord[np.isin(polymer.res_id, idxA) & (polymer.chain_id == chain_id[0])]
    domainB = polymer.coord[np.isin(polymer.res_id, idxB) & (polymer.chain_id == chain_id[1])]
    center_a = np.mean(domainA, axis=0)
    center_b = np.mean(domainB, axis=0)
    return np.sqrt(np.sum((center_a - center_b)**2))

def compute_distance_no_chain(polymer, idxA, idxB):
    """Compute the distances between two domains when there is no chain information"""
    domainA = polymer.coord[np.isin(polymer.res_id, idxA)]
    domainB = polymer.coord[np.isin(polymer.res_id, idxB)]
    center_a = np.mean(domainA, axis=0)
    center_b = np.mean(domainB, axis=0)
    return np.sqrt(np.sum((center_a - center_b)**2))

def compute_distribution_distances(path, idxA, idxB, predicted=False, chain_id = ["A", "B"], chain_information=True):
    """ Computes the distances for all the structures present in the folder path """
    all_distances = []
    start = 1
    end = 10001
    if predicted:
        start = 0
        end = 10000
        
    res_ids = [i for i in range(1, 504)]
    for i in range(start, end):
        if not predicted:
            try:
                path_struct = path + "test_"+str(i)+ ".pdb"
                pol = Polymer.from_pdb(path_struct)
            except:
                path_struct = path + "short_"+str(i+1)+ ".pdb"
                pol = Polymer.from_pdb(path_struct)

            pol.chain_id[:503] = "A"
            pol.chain_id[503:] = "B"
            pol.res_id[pol.chain_id == "A"] = res_ids
            pol.res_id[pol.chain_id == "B"] = res_ids
        else:
            try:
                path_struct = path + "structure_z_"+str(i)+ ".pdb"
                pol = Polymer.from_pdb(path_struct)
            except:
                path_struct = path + "short_"+str(i+1)+ ".pdb"
                pol = Polymer.from_pdb(path_struct)
            
        if chain_information:
            dist = compute_distance(pol, idxA, idxB, chain_id)
        else:
            dist = compute_distance_no_chain(pol, idxA, idxB)

        print(dist)
        all_distances.append(dist)
        
    return np.array(all_distances)



distances = compute_distribution_distances(path, [i for i in range(321, 503)], [i for i in range(824, 969)], predicted=True, chain_information=chain_information)
np.save(f"{path}/all_predicted_distances.npy", distances)