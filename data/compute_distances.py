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
args = parser_arg.parse_args()
path = args.path


def compute_distance(polymer, idxA, idxB, chain_id):
    """ Computes the distance between the two upper domains"""
    domainA = polymer.coord[np.isin(polymer.res_id, idxA) & (polymer.chain_id == chain_id[0])]
    domainB = polymer.coord[np.isin(polymer.res_id, idxB) & (polymer.chain_id == chain_id[1])]
    center_a = np.mean(domainA, axis=0)
    center_b = np.mean(domainB, axis=0)
    return np.sqrt(np.sum((center_a - center_b)**2))

def compute_distribution_distances(path, idxA, idxB, predicted=False, chain_id = ["A", "B"]):
    """ Computes the distances for all the structures present in the folder path """
    all_distances = []
    start = 1
    end = 10001
    if predicted:
        start = 0
        end = 10000
        
    res_ids = [i for i in range(1, 504)]
    for i in tqdm(range(start, end)):
        if not predicted:
            path_struct = path + "test_"+str(i)+ ".pdb"
            pol = Polymer.from_pdb(path_struct)
            print(pol.coord.shape)
            pol.chain_id[:503] = "A"
            pol.chain_id[503:] = "B"
            pol.res_id[pol.chain_id == "A"] = res_ids
            pol.res_id[pol.chain_id == "B"] = res_ids
        else:
            path_struct = path + "structure_z_"+str(i)+ ".pdb"
            pol = Polymer.from_pdb(path_struct)
            
        dist = compute_distance(pol, idxA, idxB, chain_id)
        all_distances.append(dist)
        
    return all_distances



distances = compute_distribution_distances(path, [i for i in range(321, 504)], [i for i in range(321, 504)], predicted=True)
np.save(f"{path}/all_predicted_distances.npy", distance)