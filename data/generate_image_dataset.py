import sys
import os
path = os.path.abspath("model")
sys.path.append(path)
import torch
import yaml
import utils
import warnings
import argparse
import numpy as np
from tqdm import tqdm
from Bio.PDB import PDBParser
from renderer import Renderer
from Bio import BiopythonWarning

parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--folder_experiment', type=str, required=True)
parser_arg.add_argument('--posed_structures', type=str, required=True)
args = parser_arg.parse_args()
folder_experiment = args.folder_experiment
path_posed_structures = args.posed_structures


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

renderer_no_ctf = Renderer(pixels_x, pixels_y, N_atoms=experiment_settings["N_residues"] * 3,
                    dfU=image_settings["renderer"]["dfU"], dfV=image_settings["renderer"]["dfV"],
                    dfang=image_settings["renderer"]["dfang"],
                    spherical_aberration=image_settings["renderer"]["spherical_aberration"],
                    accelerating_voltage=image_settings["renderer"]["accelerating_voltage"],
                    amplitude_contrast_ratio=image_settings["renderer"]["amplitude_contrast_ratio"],
                    device=device, use_ctf=False)


parser = PDBParser(PERMISSIVE=0)
batch_size = experiment_settings["batch_size"]
## We don't use any pose since we are using structure that are already posed
poses = torch.broadcast_to(torch.eye(3, 3, dtype=torch.float32, device=device)[None, :, :], (batch_size, 3, 3))
poses = torch.tensor(poses, dtype=torch.float32, device=device)
poses_translation = torch.broadcast_to(torch.zeros(3, dtype=torch.float32, device=device)[None,:], (batch_size, 3))
#Get the structures to convert them into 2d images
structures = [path_posed_structures + path for path in os.listdir(path_posed_structures) if ".pdb" in path]
indexes = [int(name.split("/")[-1].split(".")[0].split("_")[-1]) for name in structures]
#Keep the backbone only. Note that there is NO NEED to recenter, since we centered the structures when generating the
#posed structures, where the center of mass was computed using ALL the atoms.
sorted_structures = [utils.get_backbone(parser.get_structure("A", struct))[None, :, :] for _, struct in tqdm(sorted(zip(indexes[:200], structures[:200])))]
sorted_structures = torch.tensor(np.concatenate(sorted_structures, axis=0), dtype=torch.float32, device=device)

N = int(np.ceil(experiment_settings["N_images"]/batch_size))
all_images_no_noise = []
all_images_noise = []
all_images_no_noise_no_ctf = []
var_noise = image_settings["noise_var"]
for i in range(0,N):
    print(i)
    batch_structures = sorted_structures[i*batch_size:(i+1)*batch_size]
    batch_images = renderer.compute_x_y_values_all_atoms(batch_structures, poses, poses_translation)
    batch_images_no_ctf = renderer_no_ctf.compute_x_y_values_all_atoms(batch_structures, poses, poses_translation)
    all_images_no_noise.append(batch_images)
    all_images_no_noise_no_ctf.append(batch_images_no_ctf)
    batch_images_noisy = batch_images + torch.randn_like(batch_images)*np.sqrt(var_noise)
    all_images_noise.append(batch_images_noisy)

    #torch.save(batch_images_noisy, f"{folder_experiment}ImageDataSet_{i}")
    #torch.save(batch_images, f"{folder_experiment}ImageDataSetNoNoise_{i}")
    #torch.save(batch_images_no_ctf, f"{folder_experiment}ImageDataSetNoNoiseNoCTF_{i}")

all_images_noise = torch.concat(all_images_noise, dim=0)
all_images_no_noise = torch.concat(all_images_no_noise, dim=0)
all_images_no_noise_no_ctf = torch.concat(all_images_no_noise_no_ctf, dim=0)
power_no_noise = torch.var(all_images_no_noise, dim=(-2, -1))

snr = torch.mean(power_no_noise/var_noise)
print("Signal-to_noise ratio:", snr)

torch.save(all_images_noise, f"{folder_experiment}ImageDataSet")
torch.save(all_images_no_noise, f"{folder_experiment}ImageDataSetNoNoise")
torch.save(all_images_no_noise_no_ctf, f"{folder_experiment}ImageDataSetNoNoiseNoCTF")

