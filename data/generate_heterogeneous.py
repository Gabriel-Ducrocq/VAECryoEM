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
from Bio.PDB import PDBParser
import matplotlib.pyplot as plt
from Bio import BiopythonWarning
from gmm import Gaussian, EMAN2Grid
from convert_to_star import create_star_file
from renderer import project, get_posed_structure, apply_ctf

from pytorch3d.transforms import axis_angle_to_matrix

parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--folder_experiment', type=str, required=True)
parser_arg.add_argument('--folder_structures', type=str, required=True)
parser_arg.add_argument('--pose_rotation', type=str, required=False)
parser_arg.add_argument('--pose_translation', type=str, required=False)
parser_arg.add_argument('--homogeneous', default=True, action=argparse.BooleanOptionalAction)
parser_arg.add_argument('--structure_path', type=str, required=False)
args = parser_arg.parse_args()
folder_experiment = args.folder_experiment
folder_structures = args.folder_structures
pose_rotation = args.pose_rotation
poses_translation = args.pose_translation
is_homogeneous = args.homogeneous
if is_homogeneous:
    structure_path = args.structure_path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Get all the structure and sort their names to have them in the right order.
if not is_homogeneous:
    structures = [folder_structures + path for path in os.listdir(folder_structures) if ".pdb" in path]
    indexes = [int(name.split("/")[-1].split(".")[0].split("_")[-1]) for name in structures]
    sorted_structures = [struct for _, struct in sorted(zip(indexes, structures))]

with open(f"{folder_experiment}/parameters.yaml", "r") as file:
    experiment_settings = yaml.safe_load(file)

with open(f"{folder_experiment}/images.yaml", "r") as file:
    image_settings = yaml.safe_load(file)

N_images = experiment_settings["N_images"]
ctf_yaml = image_settings["ctf"]
apix = image_settings["apix"]
Npix = image_settings["Npix"]
sigma_gmm = image_settings["sigma_gmm"]
snr = image_settings["SNR"]
N_pose_per_structure = experiment_settings["N_pose_per_structure"]
centering_structure_path = experiment_settings["centering_structure_path"]
centering_structure = Polymer.from_pdb(centering_structure_path)

#Creating the CTF:
headers = ["dfU", "dfV", "dfang", "accelerating_voltage", "spherical_aberration", "amplitude_contrast_ratio"]
ctf_vals = [[ctf_yaml[header]]*N_images for header in headers]
ctf_vals = np.array([[Npix]*N_images] + [[apix]*N_images] + ctf_vals)

print("CTF VALS", ctf_vals.shape)
ctf = CTF(*ctf_vals, device=device)

#Creating the grid:
grid = EMAN2Grid(Npix, apix, device)

#Create poses:
if not is_homogeneous:
    N_images = len(sorted_structures)*N_pose_per_structure
else:
    N_images = experiment_settings["N_images"]

if not pose_rotation:
    axis_rotation = torch.randn((N_images, 3), device=device)
    norm_axis = torch.sqrt(torch.sum(axis_rotation**2, dim=-1))
    normalized_axis = axis_rotation/norm_axis[:, None]
    print("Min norm of rotation axis", torch.min(torch.sqrt(torch.sum(normalized_axis**2, dim=-1))))
    print("Max norm of rotation axis", torch.max(torch.sqrt(torch.sum(normalized_axis**2, dim=-1))))

    angle_rotation = torch.rand((N_images,1), device=device)*torch.pi
    plt.hist(angle_rotation[:, 0].detach().cpu().numpy())
    plt.show()

    axis_angle = normalized_axis*angle_rotation
    poses = axis_angle_to_matrix(axis_angle)
else:
    poses = torch.load(pose_rotation)

if not poses_translation:
    poses_translation = torch.zeros((N_images, 3), device=device)
else:
    poses_translation = torch.load(poses_translation)

print("Min translation", torch.min(poses_translation))
print("Max translation", torch.max(poses_translation))

#Finding the center of mass to center the protein
center_vector = np.mean(centering_structure.coord, axis=0)
## !!!!!!!!!!!!!!!!!!    BE CAREFUL I AM TRANSLATING A LITTLE BIT !!!!!!!!!!!!!!!!!!
center_vector += 0.5*apix

all_images = []
from time import time

if is_homogeneous:
    backbone = torch.tensor(centering_structure.coord - center_vector, dtype=torch.float32, device=device)
    backbone = torch.concatenate([backbone[None, :, :] for _ in range(N_pose_per_structure)], dim=0)

if not is_homogeneous:
    n_iter = len(sorted_structures)
else:
    assert N_images % N_pose_per_structure == 0, "Number of poses does not divide the number of images"
    n_iter = int(N_images/N_pose_per_structure)

poly = centering_structure
for i in tqdm(range(n_iter)):
    if not is_homogeneous:
        poly = Polymer.from_pdb(sorted_structures[i]) 
        backbone = poly.coord - center_vector
        backbone = torch.tensor(backbone, dtype=torch.float32, device=device)
        backbone = torch.concatenate([backbone[None, :, :] for _ in range(N_pose_per_structure)], dim=0)

    posed_backbones = get_posed_structure(backbone, poses[i*N_pose_per_structure:(i+1)*N_pose_per_structure], poses_translation[i*N_pose_per_structure:(i+1)*N_pose_per_structure])
    #batch_images = project(posed_backbones, torch.ones((backbone.shape[1], 1), device=device)*sigma_gmm, torch.tensor(poly.num_electron, device=device)[:, None], grid)
    batch_images = project(posed_backbones, torch.ones((backbone.shape[1], 1), device=device)*sigma_gmm, torch.ones((backbone.shape[1], 1), device=device), grid)
    #batch_ctf_corrupted_images = apply_ctf(batch_images, ctf, torch.tensor([j for j in range(i*N_pose_per_structure, (i+1)*N_pose_per_structure)], device=device))
    #plt.imshow(batch_ctf_corrupted_images[0].detach().numpy())
    #plt.show()
    #batch_ctf_corrupted_images_bis = apply_ctf_bis(batch_images, ctf, torch.tensor([j for j in range(i*N_pose_per_structure, (i+1)*N_pose_per_structure)]))
    all_images.append(batch_images.detach().cpu())
    #all_images.append(batch_ctf_corrupted_images.detach().cpu())
    #plt.imshow(batch_ctf_corrupted_images.detach().numpy()[0])
    #plt.show()
    #plt.imshow(batch_ctf_corrupted_images_bis.detach().numpy()[0])
    #plt.show()
    #rel_err = torch.abs((batch_ctf_corrupted_images - batch_ctf_corrupted_images_bis)/batch_ctf_corrupted_images)[0]
    #caped = torch.minimum(rel_err, torch.ones_like(rel_err))
    #plt.imshow(caped.detach().numpy())
    #plt.show()



all_images = torch.concat(all_images, dim=0)
print("Images shape", all_images.shape)
mean_variance = torch.mean(torch.var(all_images, dim=(-2, -1)))
print("Mean variance accross images", mean_variance)
noise_var = mean_variance/image_settings["SNR"]
print("Mean variance accross images", mean_variance)
print("Adding Gaussian noise with variance", noise_var)
torch.save(all_images, f"{folder_experiment}ImageDataSetNoNoise")
all_images += torch.randn((N_images, Npix, Npix))*torch.sqrt(noise_var)
print("Saving images in MRC format")
mrc.MRCFile.write(f"{folder_experiment}particles.mrcs", all_images.detach().cpu().numpy(), Apix=apix, is_vol=False)
print("Saving poses and ctf in star format.")
output_path = f"{folder_experiment}particles.star"
create_star_file(poses.detach().cpu().numpy(), poses_translation[:, :2].detach().cpu().numpy(), "particles.mrcs", N_images, Npix, apix, image_settings["ctf"], output_path)




