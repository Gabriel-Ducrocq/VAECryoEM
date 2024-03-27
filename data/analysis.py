import sys
import os
path = os.path.abspath("model")
sys.path.append(path)
import mrc
import yaml
import torch
import utils
import mrcfile
import renderer
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
from torch.utils.data import DataLoader
from pytorch3d.transforms import quaternion_to_axis_angle
from protein.main import rotate_residues, translate_residues

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


filter_aa = True



def analyze(yaml_setting_path, model_path, latent_path, structures_path, z):
    """
    train a VAE network
    :param yaml_setting_path: str, path the yaml containing all the details of the experiment
    :return:
    """
    _, image_translator, ctf, grid, gmm_repr, optimizer, dataset, N_epochs, batch_size, experiment_settings, latent_type, device, scheduler, base_structure = utils.parse_yaml(yaml_setting_path)
    vae = torch.load(model_path)
    vae.eval()
    all_latent_variables = []
    data_loader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False))
    if z is None:
        for batch_num, (indexes, batch_images, batch_poses, batch_poses_translation) in enumerate(data_loader):
            start = time()
            batch_images = batch_images.to(device)
            batch_poses = batch_poses.to(device)
            batch_poses_translation = batch_poses_translation.to(device)
            indexes = indexes.to(device)

            start_net = time()
            latent_variables, latent_mean, latent_std = vae.sample_latent(batch_images)
            all_latent_variables.append(latent_variables)
            print(latent_variables.shape)
            mask = vae.sample_mask(batch_images.shape[0])
            quaternions_per_domain, translations_per_domain = vae.decode(latent_variables)
            start_old = time()
            """
            #rotation_per_residue = model.utils.compute_rotations_per_residue(quaternions_per_domain, mask, device)
            end_old = time()
            start_new = time()
            rotation_per_residue = utils.compute_rotations_per_residue_einops(quaternions_per_domain, mask, device)
            end_new = time()
            translation_per_residue = utils.compute_translations_per_residue(translations_per_domain, mask)
            end_net = time()
            print("Net time:", end_net - start_net)
            start_deforming = time()
            predicted_structures = utils.deform_structure(gmm_repr.mus, translation_per_residue,
                                                               rotation_per_residue)

            for i, pred_struct in enumerate(predicted_structures):
                base_structure.coord = pred_struct.detach().cpu().numpy()
                base_structure.to_pdb(os.path.join(structures_path, f"structure_{batch_num*batch_size + i}.pdb"))



            end_deforming = time()
            print("Deforming time", end_deforming - start_deforming)

            end = time()
            print("Iteration duration:", end-start)
            """

        all_latent_variables = torch.concat(all_latent_variables, dim=0).detach().cpu().numpy()
        np.save(latent_path, all_latent_variables)
    else:
        z = np.load(z)
        z = torch.tensor(z).to(device)
        for i, latent_variables in enumerate(z):
            print("Latent variable number:", i)
            #latent_variables = latent_variables[None, :]
            latent_variables = torch.zeros((batch_size, z.shape[1]), device=device)
            latent_variables[:10] = latent_variables
            mask = vae.sample_mask(latent_variables.shape[0])
            quaternions_per_domain, translations_per_domain = vae.decode(latent_variables)
            quaternions_per_domain = quaternions_per_domain[:10]
            translations_per_domain = translations_per_domain[:10]
            start_old = time()
            #rotation_per_residue = model.utils.compute_rotations_per_residue(quaternions_per_domain, mask, device)
            end_old = time()
            start_new = time()
            rotation_per_residue = utils.compute_rotations_per_residue_einops(quaternions_per_domain, mask, device)
            end_new = time()
            translation_per_residue = utils.compute_translations_per_residue(translations_per_domain, mask)
            predicted_structures = utils.deform_structure(gmm_repr.mus, translation_per_residue,
                                                               rotation_per_residue)

            base_structure.coord = pred_struct.detach().cpu().numpy()
            base_structure.to_pdb(os.path.join(structures_path, f"structure_z_{i}.pdb"))





if __name__ == '__main__':
    parser_arg = argparse.ArgumentParser()
    parser_arg.add_argument('--experiment_yaml', type=str, required=True)
    parser_arg.add_argument("--model", type=str, required=True)
    parser_arg.add_argument("--latent_path", type=str, required=True)
    parser_arg.add_argument("--structures_path", type=str, required=True)
    parser_arg.add_argument("--z", type=str, required=False)
    args = parser_arg.parse_args()
    structures_path = args.structures_path
    latent_path = args.latent_path
    model_path = args.model
    path = args.experiment_yaml
    z = args.z
    analyze(path, model_path, latent_path, structures_path, z)






