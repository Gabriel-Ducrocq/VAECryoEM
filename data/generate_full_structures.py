import sys
import os
path = os.path.abspath("model")
sys.path.append(path)
import mrc
import yaml
import torch
import utils
import mrcfile
import argparse
import starfile
import numpy as np
from ctf import CTF
import seaborn as sns
from time import time
from tqdm import tqdm
import Bio.PDB as bpdb
from Bio.PDB import PDBIO
from polymer import Polymer
from Bio.PDB import PDBParser
import matplotlib.pyplot as plt
from dataset import ImageDataSet
from gmm import Gaussian, EMAN2Grid
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist
from biotite.structure.io.pdb import PDBFile
from pytorch3d.transforms import quaternion_to_axis_angle, quaternion_to_matrix

class ResSelect(bpdb.Select):
    def accept_residue(self, res):
        if res.get_resname() == "LBV":
            return False
        else:
            return True

def concat_and_save(tens, path):
    """
    Concatenate the lsit of tensor along the dimension 0
    :param tens: list of tensor with batch size as dim 0
    :param path: str, path to save the torch tensor
    :return: tensor of concatenated tensors
    """
    concatenated = torch.concat(tens, dim=0)
    np.save(path, concatenated.detach().numpy())
    return concatenated


def compute_traversals(z, dimensions = [0, 1, 2], numpoints=10):
    pca = PCA()
    z_pca = pca.fit_transform(z)
    all_trajectories = []
    all_trajectories_pca = []
    for dim in dimensions:
            traj_pca = graph_traversal(z_pca, dim, numpoints)
            ztraj_pca = pca.inverse_transform(traj_pca)
            nearest_points, _ = get_nearest_point(z, ztraj_pca)
            all_trajectories.append(nearest_points)
            all_trajectories_pca.append(traj_pca)
        
    return all_trajectories, all_trajectories_pca, z_pca, pca



def get_nearest_point(data, query):
    """
    Find closest point in @data to @query
    Return datapoint, index
    """
    ind = cdist(query, data).argmin(axis=1)
    return data[ind], ind

def graph_traversal(z_pca, dim, numpoints=10):
    print(z_pca.shape)
    print("dim", dim)
    z_pca_dim = z_pca[:, int(dim)]
    start = np.percentile(z_pca_dim, 5)
    stop = np.percentile(z_pca_dim, 95)
    traj_pca = np.zeros((numpoints, z_pca.shape[1]))
    traj_pca[:, dim] = np.linspace(start, stop, numpoints)
    return traj_pca



filter_aa = True



def deform_entire_structure(positions, rotation_matrix_per_residue, translation_per_residue, expansion_mask):
    rotation_matrix_per_atom = rotation_matrix_per_residue[:, expansion_mask, :, :]
    rotated_atoms = torch.einsum("baij, aj -> bai", rotation_matrix_per_atom, positions)
    print(rotation_matrix_per_residue.shape)
    print(rotated_atoms.shape)
    print(translation_per_residue.shape)
    print(np.max(expansion_mask))
    #translated_atoms = rotated_atoms + translation_per_residue[:, expansion_mask, :]
    return translated_atoms


def analyze(yaml_setting_path, model_path, structures_path, z, thinning=1, dimensions=[0, 1, 2], numpoints=10, generate_structures=False):
    """
    train a VAE network
    :param yaml_setting_path: str, path the yaml containing all the details of the experiment
    :return:
    """
    _, image_translator, ctf, grid, gmm_repr, optimizer, dataset, N_epochs, batch_size, experiment_settings, latent_type, device, scheduler, base_structure, lp_mask2d, mask_images, amortized  = utils.parse_yaml(yaml_setting_path)
    vae = torch.load(model_path)
    if amortized:
        vae.amortized = True
    else:
        vae.amortized = False

    with open(path, "r") as file:
        experiment_settings = yaml.safe_load(file)
        

    all_number_of_atoms = []
    f = PDBFile.read(experiment_settings["base_structure_path"])
    atom_arr_stack = f.get_structure()
    base_coordinates = torch.tensor(atom_arr_stack.coord, dtype=torch.float32, device=device)
    for chain_id in np.unique(atom_arr_stack.chain_id):
        chain = atom_arr_stack[:, atom_arr_stack.chain_id == chain_id]
        for res_id in np.unique(chain.res_id):
            n_atoms = chain[:, chain.res_id == res_id].shape[1]
            all_number_of_atoms.append(n_atoms)

    print("ALL NUMBER OF ATOMS LENGTH", len(all_number_of_atoms))
    expansion_mask = [i for i, n_atoms in enumerate(all_number_of_atoms) for _ in range(n_atoms)]
    vae.eval()
    all_latent_variables = []
    data_loader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4))
    if z is None:
        for batch_num, (indexes, batch_images, batch_poses, batch_poses_translation, _) in enumerate(data_loader):
            print("Batch number:", batch_num)
            print("dimensions", dimensions)
            start = time()
            batch_images = batch_images.to(device)
            batch_poses = batch_poses.to(device)
            batch_poses_translation = batch_poses_translation.to(device)
            indexes = indexes.to(device)

            start_net = time()
            batch_images = batch_images.flatten(start_dim=-2)
            latent_variables, latent_mean, latent_std = vae.sample_latent(batch_images, indexes)
            all_latent_variables.append(latent_variables)
            if batch_num == 500:
                all_latent_variables = torch.concat(all_latent_variables, dim=0).detach().cpu().numpy()
                np.save(f"{structures_path}z_cryosphere_1.npy", all_latent_variables)
                all_latent_variables = []



        all_latent_variables = torch.concat(all_latent_variables, dim=0).detach().cpu().numpy()
        np.save(f"{structures_path}z_cryosphere_2.npy", all_latent_variables)
        all_latent_variables = 0
        z1 = np.load(f"{structures_path}z_cryosphere_1.npy", all_latent_variables)
        z2 = np.load(f"{structures_path}z_cryosphere_2.npy", all_latent_variables)
        all_latent_variables = np.concatenate([z1, z2], axis=0)
    else:
        all_latent_variables = z


    print(all_latent_variables)
    all_latent_variables= torch.tensor(all_latent_variables[::thinning], dtype=torch.float32, device=device)
    latent_variables_loader = iter(DataLoader(all_latent_variables, shuffle=False, batch_size=batch_size))
    all_axis_angle = []
    all_translations = []
    for batch_num, z in enumerate(latent_variables_loader):  
        print("Batch number:", batch_num)
        print("dimensions", dimensions)
        start = time()
        mask = vae.sample_mask(z.shape[0])
        quaternions_per_domain, translations_per_domain = vae.decode(z)
        all_axis_angle.append(quaternion_to_axis_angle(quaternions_per_domain))
        print("Translations per domain:", translations_per_domain)
        all_translations.append(translations_per_domain)
        rotation_matrix_per_residue = utils.compute_rotations_per_residue_einops(quaternions_per_domain, mask, device)
        translation_per_residue = utils.compute_translations_per_residue(translations_per_domain, mask)

        print("ALL NUMBER OF ATOMS LENGTH", len(all_number_of_atoms))
        predicted_structures = deform_entire_structure(base_coordinates[0], rotation_matrix_per_residue, translation_per_residue, expansion_mask)

        for i, pred_struct in enumerate(predicted_structures):
            print("Saving structure", batch_num*batch_size + i)
            print(all_latent_variables.shape)
            atom_arr_stack.coord = pred_struct.detach().cpu().numpy()
            file = PDBFile()
            file.set_structure(atom_arr_stack.coord)
            file.write(os.path.join(structures_path, f"structure_z_{batch_num*batch_size + i}.pdb"))


    all_axis_angle = torch.concatenate(all_axis_angle, dim=0).detach().cpu().numpy()
    all_translations = torch.concatenate(all_translations, dim=0).detach().cpu().numpy()
    np.save(os.path.join(structures_path, f"all_axis_angle_predicted.npy"), all_axis_angle)
    np.save(os.path.join(structures_path, f"all_translations_predicted.npy"), all_translations)


if __name__ == '__main__':
    parser_arg = argparse.ArgumentParser()
    parser_arg.add_argument('--experiment_yaml', type=str, required=True)
    parser_arg.add_argument("--model", type=str, required=True)
    parser_arg.add_argument("--structures_path", type=str, required=True)
    parser_arg.add_argument("--z", type=str, required=False)
    parser_arg.add_argument("--thinning", type=int, required=False)
    parser_arg.add_argument("--num_points", type=int, required=False)
    parser_arg.add_argument('--dimensions','--list', nargs='+', type=int, help='<Required> PC dimensions along which we compute the trajectories. If not set, use pc 1, 2, 3', required=False)
    parser_arg.add_argument('--generate_structures', action=argparse.BooleanOptionalAction)
    args = parser_arg.parse_args()
    structures_path = args.structures_path
    thinning = args.thinning
    model_path = args.model
    num_points = args.num_points
    path = args.experiment_yaml
    dimensions = args.dimensions
    z = None
    if args.z is not None:
        z = np.load(args.z)
        
    generate_structures = args.generate_structures
    analyze(path, model_path, structures_path, z, dimensions=dimensions, generate_structures=generate_structures, thinning=thinning, numpoints=num_points)






