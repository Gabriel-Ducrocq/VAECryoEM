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
from torch.utils.data import DataLoader
from pytorch3d.transforms import quaternion_to_axis_angle



filter_aa = True



def pose_structure(yaml_setting_path, output_folder):
    """
    train a VAE network
    :param yaml_setting_path: str, path the yaml containing all the details of the experiment
    :return:
    """
    _, image_translator, ctf, grid, gmm_repr, optimizer, dataset, N_epochs, batch_size, experiment_settings, latent_type, device, scheduler, base_structure, lp_mask2d, mask_images  = utils.parse_yaml(yaml_setting_path)
    data_loader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4))
    for batch_num, (indexes, batch_images, batch_poses, batch_poses_translation) in enumerate(data_loader):
		print("Batch number:", batch_num)
		start = time()
		batch_images = batch_images.to(device)
		batch_poses = batch_poses.to(device)
		batch_poses_translation = batch_poses_translation.to(device)
		indexes = indexes.to(device)
		structs = gmm_repr.mus[None, :, :].repeat(batch_size, 1, 1)
		posed_structs = renderer.rotate_structure(structs, batch_poses)


		for i, pred_struct in enumerate(posed_structs):
			print("Saving structure", i+1)
			base_structure.coord = pred_struct.detach().cpu().numpy()
			base_structure.to_pdb(os.path.join(structures_path, f"structure_z_{batch_num*batch_size + i}.pdb"))


if __name__ == '__main__':
    parser_arg = argparse.ArgumentParser()
    parser_arg.add_argument('--experiment_yaml', type=str, required=True)
    parser_arg.add_argument('--ouput_folder', type=str, required=True)
    args = parser_arg.parse_args()
    path = args.experiment_yaml
    output_folder = args.output_folder
    analyze(path, output_folder)

