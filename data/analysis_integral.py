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
            nearest_points, indices_points = get_nearest_point(z, ztraj_pca)
            all_trajectories.append(nearest_points)
            all_trajectories_pca.append(traj_pca)
        
    return all_trajectories, all_trajectories_pca, z_pca, pca, indices_points



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



def analyze(yaml_setting_path, model_path, structures_path, z, thinning=1, dimensions=[0, 1, 2], numpoints=10, generate_structures=False, generate_images=False):
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

    if not generate_structures and not generate_images:
        if all_latent_variables.shape[-1] > 1:
            all_trajectories, all_trajectories_pca, z_pca, pca, indices_z = compute_traversals(all_latent_variables[::thinning], dimensions=dimensions, numpoints=numpoints)
            sns.set_style("white")
            for dim in dimensions[:-1]:
                os.makedirs(os.path.join(structures_path, f"pc{dim}/"), exist_ok=True)
                sns.kdeplot(x=z_pca[:, dim], y=z_pca[:, dim+1], fill=True, clip= (-5, 5))
                print("TRJACTORIES", all_trajectories_pca[dim][:, :])
                plt.scatter(x=all_trajectories_pca[dim][:, dim], y=all_trajectories_pca[dim][:, dim+1], c="red")
                plt.title("PCA of the latent space")
                plt.xlabel(f"PC {dim+1}, variance {pca.explained_variance_ratio_[dim]} ")
                plt.ylabel(f"PC {dim+2}, variance variance {pca.explained_variance_ratio_[dim+1]}")
                plt.savefig(os.path.join(structures_path, f"pc{dim}/pca.png"))
                plt.close()
                np.save(os.path.join(structures_path, f"pc{dim}/z_pc{dim}.npy"), all_trajectories)
                np.save(os.path.join(structures_path, f"pc{dim}/z_indices_pc{dim}.npy"), indices_z)
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
                    base_structure.to_pdb(os.path.join(structures_path, f"pc{dim}/structure_z_{i}.pdb"))

        else:
                os.makedirs(os.path.join(structures_path, f"pc0/"), exist_ok=True)
                all_trajectories = graph_traversal(all_latent_variables, 0, numpoints=numpoints)
                z_dim = torch.tensor(all_trajectories, dtype=torch.float32, device=device)
                mask = vae.sample_mask(z_dim.shape[0])
                quaternions_per_domain, translations_per_domain = vae.decode(z_dim)
                rotation_per_residue = utils.compute_rotations_per_residue_einops(quaternions_per_domain, mask, device)
                translation_per_residue = utils.compute_translations_per_residue(translations_per_domain, mask)
                predicted_structures = utils.deform_structure(gmm_repr.mus, translation_per_residue,
                                                                   rotation_per_residue)

                for i, pred_struct in enumerate(predicted_structures):
                    print("Saving structure", i+1, "from pc", 0)
                    base_structure.coord = pred_struct.detach().cpu().numpy()
                    base_structure.to_pdb(os.path.join(structures_path, f"pc0/structure_z_{i}.pdb"))

    elif generate_structures and not generate_images:
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
            rotation_per_residue = utils.compute_rotations_per_residue_einops(quaternions_per_domain, mask, device)
            translation_per_residue = utils.compute_translations_per_residue(translations_per_domain, mask)
            np.save(os.path.join(structures_path, f"rotation_per_residue.npy"), rotation_per_residue.detach().cpu().numpy())
            np.save(os.path.join(structures_path, f"translation_per_residue.npy"), translation_per_residue.detach().cpu().numpy())
            predicted_structures = utils.deform_structure(gmm_repr.mus, translation_per_residue,
                                                               rotation_per_residue)

            for i, pred_struct in enumerate(predicted_structures):
                print("Saving structure", batch_num*batch_size + i)
                print(all_latent_variables.shape)
                base_structure.coord = pred_struct.detach().cpu().numpy()
                base_structure.to_pdb(os.path.join(structures_path, f"structure_z_{batch_num*batch_size + i}.pdb"))


        all_axis_angle = torch.concatenate(all_axis_angle, dim=0).detach().cpu().numpy()
        all_translations = torch.concatenate(all_translations, dim=0).detach().cpu().numpy()
        np.save(os.path.join(structures_path, f"all_axis_angle_predicted.npy"), all_axis_angle)
        np.save(os.path.join(structures_path, f"all_translations_predicted.npy"), all_translations)

    else:
        all_images = []
        data_loader = tqdm(iter(DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)))
        for batch_num, (indexes, batch_images, batch_poses, batch_poses_translation, _) in enumerate(data_loader):
            print("Batch number:", batch_num)
            print("dimensions", dimensions)
            start = time()
            batch_images = batch_images.to(device)
            batch_poses = batch_poses.to(device)
            batch_poses_translation = batch_poses_translation.to(device)
            indexes = indexes.to(device)
            batch_images = batch_images.flatten(start_dim=-2)
            latent_variables, z, latent_std = vae.sample_latent(batch_images, indexes)
            all_latent_variables.append(latent_variables)
            mask = vae.sample_mask(z.shape[0])
            quaternions_per_domain, translations_per_domain = vae.decode(z)
            rotation_per_residue = utils.compute_rotations_per_residue_einops(quaternions_per_domain, mask, device)
            translation_per_residue = utils.compute_translations_per_residue(translations_per_domain, mask)
            predicted_structures = utils.deform_structure(gmm_repr.mus, translation_per_residue, rotation_per_residue)
            posed_predicted_structures = renderer.rotate_structure(predicted_structures, batch_poses)
            predicted_images = renderer.project(posed_predicted_structures, gmm_repr.sigmas, gmm_repr.amplitudes, grid)
            all_images.append(predicted_images.detach().cpu().numpy())


        all_images = np.concatenate(all_images, axis=0)
        mrc.MRCFile.write(f"{folder_experiment}particles_predicted.mrcs", all_images, Apix=dataset.apix, is_vol=False)





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
    parser_arg.add_argument('--generate_images', action=argparse.BooleanOptionalAction)
    args = parser_arg.parse_args()
    structures_path = args.structures_path
    thinning = args.thinning
    model_path = args.model
    num_points = args.num_points
    path = args.experiment_yaml
    dimensions = args.dimensions
    generate_images = args.generate_images
    z = None
    if args.z is not None:
        z = np.load(args.z)
        
    generate_structures = args.generate_structures
    analyze(path, model_path, structures_path, z, dimensions=dimensions, generate_structures=generate_structures, thinning=thinning, 
            numpoints=num_points, generate_images=generate_images)






