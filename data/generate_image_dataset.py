import os
import torch
import yaml
import numpy as np
from tqdm import tqdm
from Bio.PDB import PDBParser

import model.utils
from model.renderer import Renderer


with open("dataset/MD_simulation/images.yaml", "r") as file:
    image_settings = yaml.safe_load(file)

pixels_x = np.linspace(image_settings["image_lower_bounds"][0], image_settings["image_upper_bounds"][0],
                       num=image_settings["N_pixels_per_axis"][0]).reshape(1, -1)

pixels_y = np.linspace(image_settings["image_lower_bounds"][1], image_settings["image_upper_bounds"][1],
                       num=image_settings["N_pixels_per_axis"][1]).reshape(1, -1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

renderer = Renderer(pixels_x, pixels_y, N_atoms=1006 * 3,
                    period=image_settings["renderer"]["period"], std=1, defocus=image_settings["renderer"]["defocus"],
                    spherical_aberration=image_settings["renderer"]["spherical_aberration"],
                    accelerating_voltage=image_settings["renderer"]["accelerating_voltage"],
                    amplitude_contrast_ratio=image_settings["renderer"]["amplitude_contrast_ratio"],
                    device=device, use_ctf=image_settings["renderer"]["use_ctf"])

parser = PDBParser(PERMISSIVE=0)

poses = np.load("dataset/MD_simulation/poses.npy")
poses = torch.tensor(poses, dtype=torch.float32, device=device)
structures = ["dataset/MD_simulation/posed_structures/" + path for path in os.listdir("dataset/MD_simulation/posed_structures/") if ".pdb" in path]

indexes = [int(name.split("/")[-1].split(".")[0].split("_")[-1]) for name in structures]
sorted_structures = [model.utils.get_backbone(parser.get_structure("A", struct))[None, :, :] for _, struct in sorted(zip(indexes, structures))]
sorted_structures = torch.tensor(np.concatenate(sorted_structures, axis=0), dtype=torch.float32, device=device)
rotations_poses = torch.eye(3,3)[None, :, :]
rotations_poses = rotations_poses.repeat((100, 1, 1))
all_images_no_noise = []
all_images_noise = []
std_noise = 0.87
for i in range(100):
    print(i)
    batch_images = renderer.compute_x_y_values_all_atoms(sorted_structures[i*100:i*100+100], rotations_poses)
    all_images_no_noise.append(batch_images)
    batch_images_noisy = batch_images + torch.randn_like(batch_images)*std_noise
    all_images_noise.append(batch_images_noisy)

all_images_noise = torch.concat(all_images_noise, dim=0)
all_images_no_noise = torch.concat(all_images_no_noise, dim=0)
torch.save(all_images_noise, "dataset/MD_simulation/ImageDataSet")
torch.save(all_images_no_noise, "dataset/MD_simulation/ImageDataSetNoNoise")
