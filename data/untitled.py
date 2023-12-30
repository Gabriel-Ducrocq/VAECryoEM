import yaml
import utils
import argparse
import numpy as np
from tqdm import tqdm
from cryodrgn import mrc
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