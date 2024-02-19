import sys
import os
path = os.path.abspath("model")
sys.path.append(path)
import torch
import yaml
import utils
import argparse
from cryodrgn import mrc 
import numpy as np
from tqdm import tqdm
from Bio.PDB import PDBParser
from renderer import Renderer, RendererFourier
import matplotlib.pyplot as plt
from pytorch3d.transforms import axis_angle_to_matrix

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



### CHANGED CTF CORRUPTION TO False !!!
renderer_no_ctf = Renderer(pixels_x, pixels_y, N_atoms=experiment_settings["N_residues"] * 3,
                    dfU=image_settings["renderer"]["dfU"], dfV=image_settings["renderer"]["dfV"],
                    dfang=image_settings["renderer"]["dfang"],
                    spherical_aberration=image_settings["renderer"]["spherical_aberration"],
                    accelerating_voltage=image_settings["renderer"]["accelerating_voltage"],
                    amplitude_contrast_ratio=image_settings["renderer"]["amplitude_contrast_ratio"],
                    device=device, use_ctf=False)


rendererFourier = RendererFourier(190, device=device)


N_pix = image_settings["N_pixels_per_axis"][0]
noise_var = image_settings["noise_var"]
centering_structure_path = experiment_settings["centering_structure_path"]
parser = PDBParser(PERMISSIVE=0)
centering_structure = parser.get_structure("A", centering_structure_path)

#Create poses:
N_images = experiment_settings["N_images"]
axis_rotation = torch.randn((N_images, 3), device=device)
norm_axis = torch.sqrt(torch.sum(axis_rotation**2, dim=-1))
normalized_axis = axis_rotation/norm_axis[:, None]
print("Min norm of rotation axis", torch.min(torch.sqrt(torch.sum(normalized_axis**2, dim=-1))))
print("Max norm of rotation axis", torch.max(torch.sqrt(torch.sum(normalized_axis**2, dim=-1))))

#ADDED A 2pi factor !
angle_rotation = torch.rand((N_images,1), device=device)*torch.pi
plt.hist(angle_rotation[:, 0].detach().cpu().numpy())
plt.show()

axis_angle = normalized_axis*angle_rotation
poses = axis_angle_to_matrix(axis_angle)
#poses = torch.repeat_interleave(torch.eye(3,3)[None, :, :], 150000, 0)
poses_translation = torch.zeros((N_images, 3), device=device)


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

backbones = torch.tensor(utils.get_backbone(centering_structure) - center_vector, dtype=torch.float32, device=device)
backbones = torch.concatenate([backbones[None, :, :] for _ in range(100)]) 
all_images = []
from time import time
for i in tqdm(range(15000)):
    start = time()
    batch_images = renderer_no_ctf.compute_x_y_values_all_atoms(backbones[:10], poses[i*10:(i+1)*10], 
    					poses_translation[i*10:(i+1)*10])
    end = time()
    print("Time mine", end - start)

    #batch_images_fourier = rendererFourier.compute_fourier(backbones[:10], torch.transpose(poses[i*10:(i+1)*10],dim0=-2, dim1=-1))
    start = time()
    batch_images_fourier = rendererFourier.compute_fourier_test(backbones[:10], poses[i*10:(i+1)*10])
    end = time()
    print("Time fourier", end - start)
    #print(torch.max(torch.abs(batch_images_fourier.imag)))
    image_mine = batch_images[0].real.detach().numpy()
    image_fourier = batch_images_fourier[0].real.detach().numpy()
    #print("MAX relative error:", np.nanmax(np.abs(image_mine - image_fourier)))
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    fig.suptitle('Vertically stacked subplots')
    ax1.imshow(image_mine)
    ax2.imshow(image_fourier)
    #ax3.imshow(image_mine - image_fourier)
    plt.show()
    all_images.append(batch_images)

all_images = torch.concat(all_images, dim=0)
print(torch.mean(torch.var(all_images, dim=(-2, -1))))
torch.save(all_images, f"{folder_experiment}ImageDataSetNoNoise")
all_images += torch.randn((N_images, N_pix, N_pix), device=device)*np.sqrt(noise_var)

torch.save(all_images, f"{folder_experiment}ImageDataSet")
mrc.write("ImageDataSet.mrcs", all_images.detach().cpu().numpy(), Apix=1.0, is_vol=False)
#mrc.write("ImageDataSet.mrcs", np.transpose(all_images.detach().cpu().numpy(), axes=(0, 2, 1)), Apix=1.0, is_vol=False)





