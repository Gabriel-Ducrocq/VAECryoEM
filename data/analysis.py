import sys
import os
path = os.path.abspath("model")
sys.path.append(path)
import torch
import yaml
import utils
import argparse
import numpy as np
from tqdm import tqdm
from Bio.PDB import PDBParser
from renderer import Renderer
from dataset import ImageDataSet
from torch.utils.data import DataLoader


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


parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--folder_experiment', type=str, required=True)
args = parser_arg.parse_args()
folder_experiment = args.folder_experiment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(f"{folder_experiment}/parameters.yaml", "r") as file:
    experiment_settings = yaml.safe_load(file)

with open(f"{folder_experiment}/images.yaml", "r") as file:
    image_settings = yaml.safe_load(file)

pixels_x = np.linspace(image_settings["image_lower_bounds"][0], image_settings["image_upper_bounds"][0],
                       num=image_settings["N_pixels_per_axis"][0]).reshape(1, -1)

pixels_y = np.linspace(image_settings["image_lower_bounds"][1], image_settings["image_upper_bounds"][1],
                       num=image_settings["N_pixels_per_axis"][1]).reshape(1, -1)

renderer_no_ctf = Renderer(pixels_x, pixels_y, N_atoms=experiment_settings["N_residues"]*3,
                    period=image_settings["renderer"]["period"], std=1, defocus=image_settings["renderer"]["defocus"],
                    spherical_aberration=image_settings["renderer"]["spherical_aberration"],
                    accelerating_voltage=image_settings["renderer"]["accelerating_voltage"],
                    amplitude_contrast_ratio=image_settings["renderer"]["amplitude_contrast_ratio"],
                    device=device, use_ctf=False)

model_path = os.listdir(f"{folder_experiment}models/")
model = torch.load(f"{folder_experiment}models/{model_path[0]}", map_location=torch.device('cpu'))
model.device = "cpu"

images_path = torch.load(f"{folder_experiment}ImageDataSet")
poses = torch.load(f"{folder_experiment}poses")
dataset = ImageDataSet(experiment_settings["dataset_images_path"], experiment_settings["dataset_poses_path"])
data_loader = iter(DataLoader(dataset, batch_size=experiment_settings["batch_size"], shuffle=False))

parser = PDBParser(PERMISSIVE=0)
base_structure = utils.read_pdb(experiment_settings["base_structure_path"])
center_of_mass = utils.compute_center_of_mass(base_structure)
centered_based_structure = utils.center_protein(base_structure, center_of_mass)
atom_positions = torch.tensor(utils.get_backbone(centered_based_structure), dtype=torch.float32, device=device)
identity_pose = torch.broadcast_to(torch.eye(3,3)[None, :, :], (experiment_settings["batch_size"], 3, 3))

all_latent_mean = []
all_latent_std = []
all_rotations_per_residue = []
all_translation_per_residue = []
for i, (batch_images, batch_poses) in enumerate(data_loader):
    print("Batch number:", i)
    latent_variables, latent_mean, latent_std = model.sample_latent(batch_images)
    mask = model.sample_mask()
    quaternions_per_domain, translations_per_domain = model.decode(latent_mean)
    rotation_per_residue = utils.compute_rotations_per_residue(quaternions_per_domain, mask, device)
    translation_per_residue = utils.compute_translations_per_residue(translations_per_domain, mask)
    deformed_structures = utils.deform_structure(atom_positions, translation_per_residue,
                                                       rotation_per_residue)

    batch_predicted_images = renderer_no_ctf.compute_x_y_values_all_atoms(deformed_structures, batch_poses,
                                                                   latent_type=experiment_settings["latent_type"])
    batch_predicted_images_no_pose = renderer_no_ctf.compute_x_y_values_all_atoms(deformed_structures, identity_pose,
                                                                   latent_type=experiment_settings["latent_type"])

    np.save(f"{folder_experiment}predicted_images_no_pose_{i}.npy", batch_predicted_images_no_pose.detach().numpy())
    np.save(f"{folder_experiment}predicted_images_{i}.npy", batch_predicted_images.detach().numpy())
    all_latent_mean.append(latent_mean)
    all_latent_std.append(latent_std)
    all_rotations_per_residue.append(rotation_per_residue)
    all_translation_per_residue.append(translation_per_residue)



all_rotations_per_residue = concat_and_save(all_rotations_per_residue, f"{folder_experiment}all_rotations_per_residue.npy")
all_translation_per_residue = concat_and_save(all_translation_per_residue, f"{folder_experiment}all_translation_per_residue.npy")
all_latent_mean = concat_and_save(all_latent_mean, f"{folder_experiment}all_latent_mean.npy")
all_latent_std = concat_and_save(all_latent_std, f"{folder_experiment}all_latent_std.npy")


N_images = experiment_settings["N_images"]
#for i in range(N_images):











