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
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


import torch
from torch.utils.data import Dataset

class ImageDataSet(Dataset):
    def __init__(self, images_path, poses_path, poses_translation_path):
        """
        Create a dataset of images and poses
        :param images: torch.tensor(N_images, N_pix_x, N_pix_y) of images
        :param poses: torch.tensor(N_images, 3, 3) of rotation matrices of the pose
        """
        images = torch.load(images_path)
        poses = torch.load(poses_path)
        poses_translation = torch.load(poses_translation_path)
        assert images.shape[0] == poses.shape[0] and images.shape[0] == poses_translation.shape[0]
        self.images = images
        self.poses = poses
        self.poses_translation = poses_translation

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.poses[idx], self.poses_translation[idx]


parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--folder_experiment', type=str, required=True)
args = parser_arg.parse_args()
folder_experiment = args.folder_experiment


with open(f"{folder_experiment}/parameters.yaml", "r") as file:
    experiment_settings = yaml.safe_load(file)

with open(f"{folder_experiment}/images.yaml", "r") as file:
    image_settings = yaml.safe_load(file)


pixels_x = np.linspace(image_settings["image_lower_bounds"][0], image_settings["image_upper_bounds"][0],
                       num=image_settings["N_pixels_per_axis"][0]).reshape(1, -1)

pixels_y = np.linspace(image_settings["image_lower_bounds"][1], image_settings["image_upper_bounds"][1],
                       num=image_settings["N_pixels_per_axis"][1]).reshape(1, -1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

renderer = Renderer(pixels_x, pixels_y, N_atoms=experiment_settings["N_residues"] * 3,
                    dfU=image_settings["renderer"]["dfU"], dfV=image_settings["renderer"]["dfV"],
                    dfang=image_settings["renderer"]["dfang"],
                    spherical_aberration=image_settings["renderer"]["spherical_aberration"],
                    accelerating_voltage=image_settings["renderer"]["accelerating_voltage"],
                    amplitude_contrast_ratio=image_settings["renderer"]["amplitude_contrast_ratio"],
                    device=device, use_ctf=image_settings["renderer"]["use_ctf"])

dataset = ImageDataSet(folder_experiment + "ImageDataSetNoNoiseNoCTF", experiment_settings["dataset_poses_path"],
                       experiment_settings["dataset_poses_translation_path"])

data_loader = iter(DataLoader(dataset, batch_size=experiment_settings["batch_size"], shuffle=False))
all_images = []
for batch_images, batch_poses, batch_poses_translation in tqdm(data_loader):
    batch_ctf_corrupted_images = renderer.ctf_corrupting(batch_images)
    all_images.append(batch_ctf_corrupted_images)
    #plt.imshow(batch_ctf_corrupted_images[0].detach().numpy())
    #plt.show()

all_images_noNoise = torch.concatenate(all_images, dim=0)
all_images = all_images_noNoise + np.sqrt(image_settings["noise_var"])*torch.randn_like(all_images_noNoise)
plt.imshow(all_images[0].detach().numpy(), cmap="gray")
plt.show()
plt.imshow(all_images_noNoise[0].detach().numpy(), cmap="gray")
plt.show()

torch.save(all_images_noNoise, folder_experiment+"ImageDataSetNoNoise")
torch.save(all_images, folder_experiment+"ImageDataSet")
print(torch.mean(torch.var(all_images, dim=(-2, -1))))