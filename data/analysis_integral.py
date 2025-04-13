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
import renderer
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
from pytorch3d.transforms import rotation_6d_to_matrix
from pytorch3d.transforms import quaternion_to_axis_angle, quaternion_to_matrix

def calc_cor_loss(pred_images, gt_images, mask=None):
    """
    Compute the cross-correlation for each pair (predicted_image, true) image in a batch. And average them
    pred_images: torch.tensor(batch_size, side_shape**2) predicted images
    gt_images: torch.tensor(batch_size, side_shape**2) of true images, translated according to the poses.
    """
    print("PRED IMAGES", pred_images.shape)
    print("gt_images", gt_images.shape)
    if mask is not None:
        pred_images = mask(pred_images)
        gt_images = mask(gt_images)
        pixel_num = mask.num_masked
    else:
        pixel_num = pred_images.shape[-2] * pred_images.shape[-1]

    pred_images = torch.flatten(pred_images, start_dim=-2, end_dim=-1)
    gt_images = torch.flatten(gt_images, start_dim=-2, end_dim=-1)
    # b, h, w -> b, num_pix
    #pred_images = pred_images.flatten(start_dim=2)
    #gt_images = gt_images.flatten(start_dim=2)

    # b 
    ####### !!!!!!!!!!!!!! CHANGING THE CORRELATION LOSS INTO A REAL CORRELATION !!!!!!!!
    #dots = (pred_images * gt_images).sum(-1)
    dots = ((pred_images - pred_images.mean(-1)[:, None]) * (gt_images - gt_images.mean(-1)[:, None])).sum(-1)
    # b -> b 
    err = -dots / (gt_images.std(-1) + 1e-5) / (pred_images.std(-1) + 1e-5)
    # b -> 1 value
    err = err / pixel_num
    return err

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



def analyze(yaml_setting_path, model_path, structures_path, z, thinning=1, dimensions=[0, 1, 2], numpoints=10, generate_structures=False):
    """
    train a VAE network
    :param yaml_setting_path: str, path the yaml containing all the details of the experiment
    :return:
    """
    vae, image_translator, ctf, grid, gmm_repr, optimizer, dataset, N_epochs, batch_size, experiment_settings, latent_type, device, scheduler, _, lp_mask2d, mask_images = utils.parse_yaml(yaml_setting_path)
    vae = torch.load(model_path, weights_only=False)        
    vae.eval()
    all_latent_variables = []
    all_pose_rotation = []
    all_pose_rotation_symmetrized = []
    all_predicted_images = []
    all_predicted_images_symmetrized = []
    all_losses = []
    all_losses_symmetrized = []
    all_distances = []
    all_distances_id = []
    symmetric_rot = torch.tensor(np.array([[ 0.96450679,  0.11832506, -0.2360632 ],
       [ 0.12629749, -0.9918126 ,  0.01888687],
       [-0.23189567, -0.0480307 , -0.97155414]]), dtype=torch.float32, device=device)
    data_loader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4))
    if z is None:
        for batch_num, (indexes, batch_images, batch_poses, batch_poses_translation) in enumerate(data_loader):
            print("Batch number:", batch_num)
            print("dimensions", dimensions)
            start = time()
            batch_images = batch_images.to(device)
            batch_poses = batch_poses.to(device)
            batch_poses_translation = batch_poses_translation.to(device)
            indexes = indexes.to(device)

            start_net = time()
            batch_images_loss = batch_images
            batch_images = batch_images.flatten(start_dim=-2)
            latent_variables, predicted_rotation_pose, latent_mean, latent_std = vae.sample_latent(batch_images)
            #predicted_rotation_pose = vae.encoder_rotation(latent_mean)
            predicted_rotation_pose = latent_mean
            #print("Predicted rotation pose shape", predicted_rotation_pose.shape)
            predicted_rotation_matrix_pose = rotation_6d_to_matrix(predicted_rotation_pose)
            predicted_rotation_matrix_pose_symmetrized = torch.einsum("bij, jk -> bik", predicted_rotation_matrix_pose, symmetric_rot)
            #print("Predicted rotation pose matrix shape", predicted_rotation_matrix_pose.shape)
            all_pose_rotation.append(predicted_rotation_matrix_pose)
            all_pose_rotation_symmetrized.append(predicted_rotation_matrix_pose_symmetrized)
            all_latent_variables.append(latent_variables)

            tranpose_true = torch.transpose(batch_poses, dim0=-1, dim1=-2)
            prod_pred_true = (torch.einsum("bik, bkj -> bij", predicted_rotation_matrix_pose, tranpose_true).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) - 1)/2
            prod_pred_true[prod_pred_true < -1 ] = -1
            prod_pred_true[prod_pred_true > 1 ] = 1
            angles = torch.acos(prod_pred_true)*180/torch.pi
            all_distances.append(angles.detach().cpu().numpy())

            prod_id_true = (tranpose_true.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) - 1)/2
            prod_id_true[prod_id_true < -1 ] = -1
            prod_id_true[prod_id_true > 1 ] = 1
            angles_id = torch.acos(prod_id_true)*180/torch.pi
            all_distances_id.append(angles_id.detach().cpu().numpy())
            if batch_num == 500:
                all_latent_variables = torch.concat(all_latent_variables, dim=0).detach().cpu().numpy()
                all_pose_rotation = torch.concat(all_pose_rotation, dim=0).detach().cpu().numpy()
                all_pose_rotation_symmetrized = torch.concat(all_pose_rotation_symmetrized, dim=0).detach().cpu().numpy()
                all_distances = np.concatenate(all_distances, axis=0)
                np.save(f"{structures_path}z_cryosphere_1.npy", all_latent_variables)
                np.save(f"{structures_path}pose_rotation_matrix_1.npy", all_pose_rotation)
                np.save(f"{structures_path}pose_rotation_matrix_symmetrized_1.npy", all_pose_rotation_symmetrized)
                np.save(f"{structures_path}all_distances_1.npy", all_distances)
                np.save(f"{structures_path}all_distances_id_1.npy", all_distances_id)
                all_latent_variables = []
                all_pose_rotation = []
                all_pose_rotation_symmetrized = []
                all_distances = []
                all_distances_id = []


            predicted_structures = gmm_repr.mus[None, :, :].repeat(predicted_rotation_matrix_pose.shape[0], 1, 1)
            posed_predicted_structures = renderer.rotate_structure(predicted_structures, predicted_rotation_matrix_pose)
            posed_predicted_structures_symmetrized = renderer.rotate_structure(predicted_structures, predicted_rotation_matrix_pose_symmetrized)
            predicted_images = renderer.project(posed_predicted_structures, gmm_repr.sigmas, gmm_repr.amplitudes, grid)
            predicted_images_symmetrized = renderer.project(posed_predicted_structures_symmetrized, gmm_repr.sigmas, gmm_repr.amplitudes, grid)
            all_predicted_images.append(predicted_images.detach().cpu().numpy())
            all_predicted_images_symmetrized.append(predicted_images_symmetrized.detach().cpu().numpy())
            losses = calc_cor_loss(predicted_images, batch_images_loss, mask=None)
            all_losses.append(losses.detach().cpu().numpy())
            losses_symmetrized = calc_cor_loss(predicted_images_symmetrized, batch_images_loss, mask=None)
            print("LOSSES SHAPE", losses.shape)
            all_losses_symmetrized.append(losses_symmetrized.detach().cpu().numpy())


        all_distances = np.concatenate(all_distances_id, axis=0)
        np.save(f"{structures_path}all_distances_id_2.npy", all_distances_id)
        all_distances = np.concatenate(all_distances, axis=0)
        np.save(f"{structures_path}all_distances_2.npy", all_distances)
        all_predicted_images = np.concatenate(all_predicted_images, axis=0)
        all_predicted_images_symmetrized = np.concatenate(all_predicted_images_symmetrized, axis=0)
        all_losses = np.concatenate(all_losses, axis=0)
        all_losses_symmetrized = np.concatenate(all_losses_symmetrized, axis=0)
        np.save(f"{structures_path}all_losses.npy", all_losses)
        np.save(f"{structures_path}all_losses_symmetrized.npy", all_losses_symmetrized)
        mrc.MRCFile.write(f"{structures_path}particles_predicted.mrcs", all_predicted_images, Apix=1.0, is_vol=False)
        mrc.MRCFile.write(f"{structures_path}particles_predicted_symmetrized.mrcs", all_predicted_images_symmetrized, Apix=1.0, is_vol=False)
        all_latent_variables = torch.concat(all_latent_variables, dim=0).detach().cpu().numpy()
        all_pose_rotation = torch.concat(all_pose_rotation, dim=0).detach().cpu().numpy()
        np.save(f"{structures_path}z_cryosphere_2.npy", all_latent_variables)
        np.save(f"{structures_path}pose_rotation_matrix_2.npy", all_pose_rotation)
        all_latent_variables = 0
        z1 = np.load(f"{structures_path}z_cryosphere_1.npy", all_latent_variables)
        z2 = np.load(f"{structures_path}z_cryosphere_2.npy", all_latent_variables)
        all_latent_variables = np.concatenate([z1, z2], axis=0)
        rotation_matrices_1 = np.load(f"{structures_path}pose_rotation_matrix_1.npy")
        rotation_matrices_2 = np.load(f"{structures_path}pose_rotation_matrix_2.npy")
        all_rotation_matrices = np.concatenate([rotation_matrices_1, rotation_matrices_2], axis=0)
    else:
        all_latent_variables = z




    if not generate_structures:
        if all_latent_variables.shape[-1] > 1:
            all_trajectories, all_trajectories_pca, z_pca, pca = compute_traversals(all_latent_variables[::thinning], dimensions=dimensions, numpoints=numpoints)
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

    else:
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






