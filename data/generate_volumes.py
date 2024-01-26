import sys
import os
path = os.path.abspath("model")
sys.path.append(path)
import yaml
import torch
import utils
import argparse
import mrcfile
import numpy as np
from tqdm import tqdm
from cryodrgn import mrc
from Bio.PDB import PDBParser
from renderer import Renderer
from Bio import BiopythonWarning

parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--folder_experiment', type=str, required=True)
parser_arg.add_argument('--folder_volumes', type=str, required=True)
parser_arg.add_argument('--folder_structures', type=str, required=True)
parser_arg.add_argument('--batch_size', type=str, required=True)
args = parser_arg.parse_args()
folder_experiment = args.folder_experiment
folder_volumes = args.folder_volumes
folder_structures = args.folder_structures
batch_size = args.batch_size


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
                    device=device, use_ctf=False)


parser = PDBParser(PERMISSIVE=0)
batch_size = experiment_settings["batch_size"]
## We don't use any pose since we are using structure that are already posed
poses = torch.eye(3, 3, dtype=torch.float32, device=device)[None, :, :]
poses_translation = torch.zeros(3, dtype=torch.float32, device=device)[None,:]
#Get the structures to convert them into 2d images
structures = [folder_structures + path for path in os.listdir(folder_structures) if ".pdb" in path]
#Keep the backbone only. Note that there is NO NEED to recenter, since we centered the structures when generating the
#posed structures, where the center of mass was computed using ALL the atoms.
centering_structure_path = experiment_settings["centering_structure_path"]
centering_structure = parser.get_structure("A", centering_structure_path)
center_vector = utils.compute_center_of_mass(centering_structure)
indexes = [int(name.split("/")[-1].split(".")[0].split("_")[-1]) for name in structures]
print(indexes)
all_structures = [utils.get_backbone(utils.center_protein(parser.get_structure("A", struct),center_vector[0]))[None, :, :] for struct in tqdm(structures)]
all_structures
all_structures = torch.tensor(np.concatenate(all_structures, axis=0), dtype=torch.float32, device=device)
N = len(all_structures)
for i in tqdm(range(0,N)):
    batch_structures = all_structures[i]
    batch_volumes = renderer.compute_x_y_values_all_atoms(batch_structures, poses, poses_translation, volume=True)
    mrc.write(f"{folder_volumes}volume_{indexes[i]}.mrc", np.transpose(batch_volumes[0].detach().cpu().numpy(), axes=(2, 1, 0)), Apix=1.0, is_vol=True)

    print("\n\n\n")
    #with mrcfile.new(f"{folder_volumes}volume_{i}.mrc", overwrite=True) as mrc:
    #	print(batch_volumes.shape)
    # 	print(np.sum(batch_volumes[0,:, :, 0].detach().numpy()))
    #	mrc.set_data(batch_volumes[0,:, :, :].detach().numpy())
    #	mrc.header["origin"]["x"] = -95
    #	mrc.header["origin"]["y"] = -95
    #	mrc.header["origin"]["z"] = -95

    #	mrc.header["nxstart"] = -95
    #	mrc.header["nystart"] = -95
    #	mrc.header["nzstart"] = -95
    #	print(mrc.header["origin"])




