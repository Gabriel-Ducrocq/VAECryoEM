import os
import sys
path = os.path.abspath("model")
sys.path.append(path)
import torch
import yaml
import utils
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from cryodrgn import mrc 
from Bio.PDB import PDBParser
import matplotlib.pyplot as plt
from renderer import Renderer, RendererFourier
from pytorch3d.transforms import axis_angle_to_matrix

parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--folder_experiment', type=str, required=True)
parser_arg.add_argument('--folder_structures', type=str, required=True)
parser_arg.add_aegument('--pose_rotation', type=str, required=False)
parser_arg.add_aegument('--pose_translation', type=str, required=False)
args = parser_arg.parse_args()
folder_experiment = args.folder_experiment
folder_structures = args.folder_structures
pose_rotation = args.pose_rotation
poses_translation = args.poses_translation


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Get all the structure and sort their names to have them in the right order.
structures = [folder_structures + path for path in os.listdir(folder_structures) if ".pdb" in path]
indexes = [int(name.split("/")[-1].split(".")[0].split("_")[-1]) for name in structures]
sorted_structures = [struct for _, struct in sorted(zip(indexes, structures))]

with open(f"{folder_experiment}/parameters.yaml", "r") as file:
    experiment_settings = yaml.safe_load(file)

with open(f"{folder_experiment}/images.yaml", "r") as file:
    image_settings = yaml.safe_load(file)

pixels_x = np.linspace(image_settings["image_lower_bounds"][0], image_settings["image_upper_bounds"][0],
                       num=image_settings["N_pixels_per_axis"][0]).reshape(1, -1)

pixels_y = np.linspace(image_settings["image_lower_bounds"][1], image_settings["image_upper_bounds"][1],
                       num=image_settings["N_pixels_per_axis"][1]).reshape(1, -1)


renderer = Renderer(pixels_x, pixels_y, N_atoms=experiment_settings["N_residues"] * 3,
                    dfU=image_settings["renderer"]["dfU"], dfV=image_settings["renderer"]["dfV"],
                    dfang=image_settings["renderer"]["dfang"],
                    spherical_aberration=image_settings["renderer"]["spherical_aberration"],
                    accelerating_voltage=image_settings["renderer"]["accelerating_voltage"],
                    amplitude_contrast_ratio=image_settings["renderer"]["amplitude_contrast_ratio"],
                    device=device, use_ctf=image_settings["renderer"]["use_ctf"], std = image_settings["renderer"]["std_volume"] if "std_volume" in image_settings["renderer"] else 1)


rendererFourier = RendererFourier(190, device=device)


N_pose_per_structure = experiment_settings["N_pose_per_structure"]
N_pix = image_settings["N_pixels_per_axis"][0]
noise_var = image_settings["noise_var"]
centering_structure_path = experiment_settings["centering_structure_path"]
parser = PDBParser(PERMISSIVE=0)
centering_structure = parser.get_structure("A", centering_structure_path)

#Create poses:
N_images = len(sorted_structures)*N_pose_per_structure
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
    #poses = torch.repeat_interleave(torch.eye(3,3)[None, :, :], 150000, 0)
else:
    poses = torch.load(pose_rotation)

if not poses_translation:
    poses_translation = torch.rand((N_images, 3), device=device)*20 - 10
    #poses_translation = torch.zeros((N_images, 3), device=device)
else:
    poses_translation = torch.load(poses_translation)


poses_py = poses.detach().cpu().numpy()
poses_translation_py = poses_translation.detach().cpu().numpy()


print("Min translation", torch.min(poses_translation))
print("Max translation", torch.max(poses_translation))

np.save(f"{folder_experiment}poses.npy", poses_py)
np.save(f"{folder_experiment}poses_translation.npy", poses_translation_py)
torch.save(poses, f"{folder_experiment}poses")
torch.save(poses_translation, f"{folder_experiment}poses_translation")

#Finding the center of mass to center the protein
center_vector = utils.compute_center_of_mass(centering_structure)

all_images = []
from time import time
for i in tqdm(range(len(sorted_structures))):
	structure = parser.get_structure("A", sorted_structures[i])
	backbone = utils.get_backbone(structure) - center_vector
	backbone = torch.tensor(backbone, dtype=torch.float32, device=device)
	backbone = torch.concatenate([backbone[None, :, :] for _ in range(N_pose_per_structure)], dim=0)
	batch_images = renderer.compute_x_y_values_all_atoms(backbone, poses[i*N_pose_per_structure:(i+1)*N_pose_per_structure], poses_translation[i*N_pose_per_structure:(i+1)*N_pose_per_structure])
	plt.imshow(batch_images[0].detach().cpu().numpy())
	plt.show()
	all_images.append(batch_images.detach().cpu())

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
import os
import sys
path = os.path.abspath("model")
sys.path.append(path)
import yaml
import utils
import torch
import pickle
import warnings
import argparse
import utils_data
import numpy as np
from tqdm import tqdm
from cryodrgn import mrc  
from Bio.PDB import PDBParser
from renderer import Renderer
import matplotlib.pyplot as plt
from Bio import BiopythonWarning
from pytorch3d.transforms import axis_angle_to_matrix

parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--folder_experiment', type=str, required=True)
parser_arg.add_argument('--folder_structures', type=str, required=True)
parser_arg.add_argument('--Apix', type=float, required=True)
args = parser_arg.parse_args()
folder_experiment = args.folder_experiment
folder_structures = args.folder_structures
Apix = args.Apix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(f"{folder_experiment}/parameters.yaml", "r") as file:
    experiment_settings = yaml.safe_load(file)

with open(f"{folder_experiment}/images.yaml", "r") as file:
    image_settings = yaml.safe_load(file)

pixels_x = np.linspace(image_settings["image_lower_bounds"][0], image_settings["image_upper_bounds"][0],
                       num=image_settings["N_pixels_per_axis"][0]).reshape(1, -1)

pixels_y = np.linspace(image_settings["image_lower_bounds"][1], image_settings["image_upper_bounds"][1],
                       num=image_settings["N_pixels_per_axis"][1]).reshape(1, -1)


renderer = Renderer(pixels_x, pixels_y, N_atoms=experiment_settings["N_residues"] * 3,
                    dfU=image_settings["renderer"]["dfU"], dfV=image_settings["renderer"]["dfV"],
                    dfang=image_settings["renderer"]["dfang"],
                    spherical_aberration=image_settings["renderer"]["spherical_aberration"],
                    accelerating_voltage=image_settings["renderer"]["accelerating_voltage"],
                    amplitude_contrast_ratio=image_settings["renderer"]["amplitude_contrast_ratio"],
                    device=device, use_ctf=image_settings["renderer"]["use_ctf"])

N_pose_per_struct = experiment_settings["N_pose_per_structure"]
N_pix = image_settings["N_pixels_per_axis"][0]
centering_structure_path = experiment_settings["centering_structure_path"]
parser = PDBParser(PERMISSIVE=0)
centering_structure = parser.get_structure("A", centering_structure_path)

#Get all the structure and sort their names to have them in the right order.
print("N_files in folder structure", len(os.listdir(folder_structures)))
structures = [folder_structures + path for path in os.listdir(folder_structures) if ".pdb" in path]
indexes = [int(name.split("/")[-1].split(".")[0].split("_")[-1]) for name in structures]
sorted_structures = [struct for _, struct in sorted(zip(indexes, structures))]


#Create poses:
N_images = experiment_settings["N_images"]*N_pose_per_struct
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
poses_translation = torch.zeros((N_images, 3), device=device)

poses_py = poses.detach().cpu().numpy()
poses_translation_py = poses_translation.detach().cpu().numpy()


print("Min translation", torch.min(poses_translation))
print("Max translation", torch.max(poses_translation))

np.save(f"{folder_experiment}poses.npy", poses_py)
np.save(f"{folder_experiment}poses_translation.npy", poses_translation_py)
torch.save(poses, f"{folder_experiment}poses")
torch.save(poses_translation, f"{folder_experiment}poses_translation")

 
center_vector = utils.compute_center_of_mass(centering_structure)
all_images = []
import matplotlib.pyplot as plt
print("LEN STRUCTURES", len(sorted_structures))
centered_structure = utils.center_protein(centering_structure, center_vector[0])
backbone = utils.get_backbone(centered_structure)[None, :, :]
backbone = torch.tensor(backbone, dtype=torch.float32, device=device)
backbones = torch.concatenate([backbone for _ in range(N_pose_per_struct)], dim=0)
with warnings.catch_warnings():
	warnings.simplefilter('ignore', BiopythonWarning)
	for i, structure in tqdm(enumerate(sorted_structures)):
		#posed_structure = utils_data.compute_poses(structure, poses_py[i], poses_translation_py[i], center_vector)
		batch_images = renderer.compute_x_y_values_all_atoms(backbones, poses[i*N_pose_per_struct:(i+1)*N_pose_per_struct], poses_translation[i*N_pose_per_struct:(i+1)*N_pose_per_struct])
		all_images.append(batch_images)


all_images = torch.concatenate(all_images, dim=0)
torch.save(all_images, f"{folder_experiment}ImageDataSetNoNoise")
mean_variance_signal = torch.mean(torch.var(all_images, dim=(-2, -1)))
noise_var = mean_variance_signal*10
print(f"Mean variance of non noisy images: {mean_variance_signal}, adding noise with variance {noise_var}.")
all_images += torch.randn((N_images, N_pix, N_pix), device=device)*torch.sqrt(noise_var)
torch.save(all_images, f"{folder_experiment}ImageDataSet")
all_images_np = np.transpose(all_images.detach().cpu().numpy(), axes=(0, 2, 1))
mrc.write(f"{folder_experiment}ImageDataSet.mrcs", all_images.detach().cpu().numpy(), Apix=Apix, is_vol=False)
with open(f"{folder_experiment}poses.pkl", "wb") as f:
	pickle.dump((poses_py, poses_translation_py[:, :2]), f)

np.save(f"{folder_experiment}ExcerptImageDataSetNoNoise", all_images.detach().cpu().numpy()[1:N_pose_per_struct*10:N_pose_per_struct])
mrc.write(f"{folder_experiment}ExcerptImageDataSet.mrcs", all_images.detach().cpu().numpy()[1:N_pose_per_struct*10:N_pose_per_struct], Apix=Apix, is_vol=False)

"""

