import os
import torch
import utils
import numpy as np
from tqdm import tqdm
from Bio.PDB import PDBParser

parser = PDBParser(PERMISSIVE=0)

poses = torch.load("../../VAEProtein/data/vaeContinuousMD/training_rotations_matrices")
poses = poses.numpy()
structures = ["../../VAEProtein/data/MD_dataset/" + path for path in os.listdir("../../VAEProtein/data/MD_dataset/") if ".pdb" in path]

indexes = [int(name.split("/")[-1].split(".")[0].split("_")[-1]) for name in structures]
sorted_structures = [struct for _, struct in sorted(zip(indexes, structures))]

center_vector = utils.compute_center_of_mass(parser.get_structure("A", sorted_structures[0]))
for i, structure in tqdm(enumerate(sorted_structures)):
    if i%100 == 0:
        print(i)

    posed_structure = utils.compute_poses(structure, poses[i], center_vector)
    utils.save_structure(posed_structure, f"dataset/MD_simulation/posed_structures/structure_{i+1}.pdb")
    np.save("dataset/MD_simulation/poses.npy", poses)
