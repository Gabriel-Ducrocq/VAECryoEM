import os
import sys
import mrc
path = os.path.abspath("model")
sys.path.append(path)
import utils as utils_model
import yaml
import torch
import pickle
import polymer
import warnings
import argparse
import numpy as np
from ctf import CTF
from tqdm import tqdm
#from cryodrgn import mrc
import utils_data as utils
from Bio.PDB import PDBParser
import matplotlib.pyplot as plt
from Bio import BiopythonWarning
from pytorch3d.transforms import axis_angle_to_matrix
from renderer import project, rotate_structure, apply_ctf

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
from pytorch3d.transforms import axis_angle_to_matrix
from renderer import project, rotate_structure, apply_ctf



parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--base_structure', type=str, required=True)
parser_arg.add_argument('--chain_id', type=str, required=True)
parser_arg.add_argument('--residue_start', type=int, required=True)
parser_arg.add_argument('--residue_end', type=int, required=True)
parser_arg.add_argument('--folder_experiment', type=str, required=True)
parser_arg.add_argument('--N_struct', type=int, required=True)
parser_arg.add_argument('--N_pose_per_struct', type=int, required=True)
args = parser_arg.parse_args()
base_structure_path = args.base_structure
residue_start = args.residue_start
residue_end = args.residue_end
chain_id = args.chain_id
folder_experiment = args.folder_experiment
N_struct = args.N_struct
N_pose_per_structure = args.N_pose_per_struct
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(f"{folder_experiment}/parameters.yaml", "r") as file:
    experiment_settings = yaml.safe_load(file)

with open(f"{folder_experiment}/images.yaml", "r") as file:
    image_settings = yaml.safe_load(file)


sigma_gmm = image_settings["sigma_gmm"]
apix = image_settings["apix"]
Npix = image_settings["Npix"]
N_images = N_struct*N_pose_per_structure
ctf_yaml = image_settings["ctf"]
headers = ["dfU", "dfV", "dfang", "accelerating_voltage", "spherical_aberration", "amplitude_contrast_ratio"]
ctf_vals = [[ctf_yaml[header]]*N_images for header in headers]
ctf_vals = np.array([[Npix]*N_images] + [[apix]*N_images] + ctf_vals)

print("CTF VALS", ctf_vals.shape)
ctf = CTF(*ctf_vals, device=device)
grid = EMAN2Grid(Npix, apix, device)
image_translator = utils.SpatialGridTranslate(D=Npix, device=device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = PDBParser(PERMISSIVE=0)

#Generate rotations for conformations
rotation_axis = np.array([[0, 1, 0] for _ in range(N_struct)])
rotation_angle = np.zeros((N_struct, 1))
rotation_angle[:int(N_struct/2), :] = np.random.normal(size=(int(N_struct/2), 1))*0.2 +np.pi/3
rotation_angle[int(N_struct/2):, :] = np.random.normal(size=(int(N_struct/2), 1))*0.2 +2*np.pi/3

rotation_angle[:, :] = np.linspace(0, 2*np.pi, N_struct)[:, None]

axis_angle = torch.tensor(rotation_angle*rotation_axis, dtype=torch.float32, device=device)
conformation_matrix_torch = axis_angle_to_matrix(axis_angle)
conformation_matrix_np = conformation_matrix_torch.detach().cpu().numpy()

plt.hist(rotation_angle[:, 0])
plt.show()

np.save(f"{folder_experiment}ground_truth/rotation_axis_conformations.npy", rotation_axis)
np.save(f"{folder_experiment}ground_truth/rotation_angle.npy", rotation_angle)

#Generate poses
axis_rotation_poses = torch.randn((N_images, 3), device=device)
norm_axis = torch.sqrt(torch.sum(axis_rotation_poses**2, dim=-1))
normalized_axis = axis_rotation_poses/norm_axis[:, None]
print("Min norm of rotation axis", torch.min(torch.sqrt(torch.sum(normalized_axis**2, dim=-1))))
print("Max norm of rotation axis", torch.max(torch.sqrt(torch.sum(normalized_axis**2, dim=-1))))

angle_rotation_poses = torch.rand((N_images,1), device=device)*torch.pi
plt.hist(angle_rotation_poses[:, 0].detach().cpu().numpy())
plt.show()

axis_angle_poses = normalized_axis*angle_rotation_poses
poses = axis_angle_to_matrix(axis_angle_poses)
#poses = torch.repeat_interleave(torch.eye(3,3)[None, :, :], 150000, 0)
poses_translation = torch.rand((N_images, 3), device=device)*20 - 10
##################################################             TRANSLATIONS ARE SET TO ZEROS #######################
poses_translation = torch.zeros_like(poses_translation, dtype=torch.float32, device=device)
poses_translation = poses_translation[:, :2]
shiftX = poses_translation[:, 0] /apix
shiftY = poses_translation[:, 1] /apix
#poses_translation = torch.zeros((N_images, 3), device=device)


poses_py = poses.detach().cpu().numpy()
poses_translation_py = poses_translation.detach().cpu().numpy()


print("Min translation", torch.min(poses_translation))
print("Max translation", torch.max(poses_translation))

np.save(f"{folder_experiment}poses.npy", poses_py)
np.save(f"{folder_experiment}poses_translation.npy", poses_translation_py)
torch.save(poses, f"{folder_experiment}poses")
torch.save(poses_translation, f"{folder_experiment}poses_translation")

size_prot = []
faulty_indexes = []
all_images = []
for i in tqdm(range(N_struct)):
	base_structure = polymer.Polymer.from_pdb(base_structure_path)
	if len(base_structure) not in size_prot:
		size_prot.append(len(base_structure))
		faulty_indexes.append(i)

	center_vector = np.mean(base_structure.coord, axis=0)
	base_structure.coord = base_structure.coord - center_vector
	#backbone = base_structure.coord - center_vector
	#Saving the generated structure.
	base_structure.coord[(base_structure.chain_id==chain_id) & (base_structure.res_id >= residue_start) & (base_structure.res_id <=residue_end)] = np.einsum("n m, l m -> l n", conformation_matrix_np[i],
																		base_structure.coord[(base_structure.chain_id==chain_id) & (base_structure.res_id >= residue_start) & (base_structure.res_id <=residue_end)])

	base_structure.to_pdb(f"{folder_experiment}ground_truth/structures/structure_{i+1}.pdb")
	#utils.rotate_domain_pdb_structure(struct_centered, residue_start, residue_end, conformation_matrix_np[i])
	#utils.save_structure(struct_centered, f"{folder_experiment}ground_truth/structures/structure_{i+1}.pdb")

	#backbone_torch = torch.tensor(backbone, dtype=torch.float32, device=device)
	backbone_torch = torch.tensor(base_structure.coord, dtype=torch.float32, device=device)
	backbone_torch = torch.concatenate([backbone_torch[None, :, :] for _ in range(N_pose_per_structure)], dim=0)
	# Duplicating the deformed backbone and projecting it.
	amplitudes = torch.tensor(base_structure.num_electron, dtype=torch.float32, device=device)[:, None]
	posed_backbones = rotate_structure(backbone_torch, poses[i*N_pose_per_structure:(i+1)*N_pose_per_structure])
	batch_images = project(posed_backbones, torch.ones((backbone_torch.shape[1], 1), device=device)*sigma_gmm, amplitudes, grid)
	batch_ctf_corrupted_images = apply_ctf(batch_images, ctf, torch.tensor([j for j in range(i*N_pose_per_structure, (i+1)*N_pose_per_structure)], device=device))
	#batch_ctf_corrupted_images = batch_images
	batch_poses_translation = - poses_translation[i*N_pose_per_structure:(i+1)*N_pose_per_structure]
	batch_translated_images = image_translator.transform(batch_ctf_corrupted_images, batch_poses_translation[:, None, :])
	all_images.append(batch_translated_images.detach().cpu())
	#Deforming the structure once for 15 poses
	#backbone_torch[residue_start*3:residue_end*3, :] = torch.transpose(torch.matmul(conformation_matrix_torch[i], torch.transpose(backbone_torch[residue_start*3:residue_end*3, :], dim0=0, dim1=1)), 
	#															dim0=0, dim1=1)


	#batch_images = renderer.compute_x_y_values_all_atoms(backbone_torch, poses[i*N_pose_per_structure:(i+1)*N_pose_per_structure], poses_translation[i*N_pose_per_structure:(i+1)*N_pose_per_structure])
	#plt.imshow(batch_images[0].detach().cpu().numpy())
	#plt.show()
	#all_images.append(batch_images.detach().cpu())



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
print(size_prot)
if len(size_prot) > 1:
    print("Some proteins have different residue numbers :", faulty_indexes)

mrc.MRCFile.write(f"{folder_experiment}particles.mrcs", all_images.detach().cpu().numpy(), Apix=apix, is_vol=False)
print("Saving poses and ctf in star format.")
output_path = f"{folder_experiment}particles.star"
create_star_file(poses.detach().cpu().numpy(), shiftX[:, None].detach().cpu().numpy(), shiftY[:, None].detach().cpu().numpy(), "particles.mrcs",
 N_images, Npix, apix, image_settings["ctf"], output_path)

"""
all_images = torch.concat(all_images, dim=0)
mean_variance = torch.mean(torch.var(all_images, dim=(-2, -1)))
print("Mean variance accross images", mean_variance)
noise_var = 10*mean_variance
print("Adding Gaussian noise with variance", noise_var)
torch.save(all_images, f"{folder_experiment}ImageDataSetNoNoise")
#all_images += torch.randn((N_images, N_pix, N_pix), device=device)*torch.sqrt(noise_var)
all_images += torch.randn((N_images, N_pix, N_pix))*torch.sqrt(noise_var)

torch.save(all_images, f"{folder_experiment}ImageDataSet")
mrc.write(f"{folder_experiment}ImageDataSet.mrcs", np.transpose(all_images.detach().cpu().numpy(), axes=(0, 2, 1)), Apix=1.0, is_vol=False)
with open(f"{folder_experiment}poses.pkl", "wb") as f:
	pickle.dump((poses_py, poses_translation_py[:, :2]), f)
"""
