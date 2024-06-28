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
from time import time
from tqdm import tqdm
import Bio.PDB as bpdb
from Bio.PDB import PDBIO
from polymer import Polymer
from Bio.PDB import PDBParser
from dataset import ImageDataSet
from gmm import Gaussian, EMAN2Grid
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist
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


def compute_traversals(z, dimensions = [0, 1, 2], numpoints=10, compound=False):
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
        
    return all_trajectories, all_trajectories_pca



def get_nearest_point(data, query):
    """
    Find closest point in @data to @query
    Return datapoint, index
    """
    ind = cdist(query, data).argmin(axis=1)
    return data[ind], ind

def graph_traversal(z_pca, dim, numpoints=10):
    z_pca_dim = z_pca[:, dim]
    start = np.percentile(z_pca_dim, 5)
    stop = np.percentile(z_pca_dim, 95)
    traj_pca = np.zeros((numpoints, z_pca.shape[1]))
    traj_pca[:, dim] = np.linspace(start, stop, numpoints)
    return traj_pca



filter_aa = True



def analyze(yaml_setting_path, model_path, structures_path, z, thinning=10, dimensions=[0, 1, 2], numpoints=10):
    """
    train a VAE network
    :param yaml_setting_path: str, path the yaml containing all the details of the experiment
    :return:
    """
    _, image_translator, ctf, grid, gmm_repr, optimizer, dataset, N_epochs, batch_size, experiment_settings, latent_type, device, scheduler, base_structure, lp_mask2d, mask_images  = utils.parse_yaml(yaml_setting_path)
    vae = torch.load(model_path)
    vae.eval()
    all_latent_variables = []
    data_loader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4))
    if z is None:
        for batch_num, (indexes, batch_images, batch_poses, batch_poses_translation) in enumerate(data_loader):
            print("Batch number:", batch_num)
            start = time()
            batch_images = batch_images.to(device)
            batch_poses = batch_poses.to(device)
            batch_poses_translation = batch_poses_translation.to(device)
            indexes = indexes.to(device)

            start_net = time()
            batch_images = batch_images.flatten(start_dim=-2)
            latent_variables, latent_mean, latent_std = vae.sample_latent(batch_images)
            all_latent_variables.append(latent_variables)


        all_latent_variables = torch.concat(all_latent_variables, dim=0).detach().cpu().numpy()
        np.save(f"{structures_path}z_cryosphere.npy", all_latent_variables)
    else:
        all_latent_variables = z

    all_trajectories, all_trajectories_pca = compute_traversals(all_latent_variables[::thinning], dimensions=dimensions, numpoints=numpoints)
    for dim in dimensions:
        z_dim = torch.tensor(all_trajectories[dim], dtype=torch.float32, device=device)
        mask = vae.sample_mask(z_dim.shape[0])
        quaternions_per_domain, translations_per_domain = vae.decode(z_dim)
        rotation_per_residue = utils.compute_rotations_per_residue_einops(quaternions_per_domain, mask, device)
        translation_per_residue = utils.compute_translations_per_residue(translations_per_domain, mask)
        predicted_structures = utils.deform_structure(gmm_repr.mus, translation_per_residue,
                                                           rotation_per_residue)

        for i, pred_struct in enumerate(predicted_structures):
            print("Saving structure", i+1, "from pc", dim)
            base_structure.coord = pred_struct.detach().cpu().numpy()
            base_structure.to_pdb(os.path.join(structures_path, f"pc{i}/structure_z_{i}.pdb"))



if __name__ == '__main__':
    parser_arg = argparse.ArgumentParser()
    parser_arg.add_argument('--experiment_yaml', type=str, required=True)
    parser_arg.add_argument("--model", type=str, required=True)
    parser_arg.add_argument("--structures_path", type=str, required=True)
    parser_arg.add_argument("--z", type=str, required=False)
    parser_arg.add_argument("--thinning", type=int, required=False)
    parser_arg.add_argument("--num_points", type=int, required=False)
    parser_arg.add_argument('--dimensions','--list', nargs='+', help='<Required> PC dimensions along which we compute the trajectories. If not set, use pc 1, 2, 3', required=False)
    args = parser_arg.parse_args()
    structures_path = args.structures_path
    model_path = args.model
    path = args.experiment_yaml
    z = args.z
    analyze(path, model_path, structures_path, z)






