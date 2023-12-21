import numpy as np
import pickle
import yaml
import argparse

parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--yaml', type=str, required=True)
parser_arg.add_argument('--pkl', type=str, required=True)
parser_arg.add_argument('--Nimages', type=int, required=True)
args = parser_arg.parse_args()

with open(args.yaml, "r") as file:
	image_settings = yaml.safe_load(file)

ctf_params = np.zeros((args.Nimages, 9))
ctf_params[:, 0] = image_settings["N_pixels_per_axis"][0]
ctf_params[:, 1] = (image_settings["image_upper_bounds"][0] - image_settings["image_lower_bounds"][0])/image_settings["N_pixels_per_axis"][0]
ctf_params[:, 2] = image_settings["dfU"]
ctf_params[:, 3] = image_settings["dfV"]
ctf_params[:, 4] = image_settings["dfang"]
ctf_params[:, 5] = image_settings["accelerating_voltage"]
ctf_params[:, 6] = image_settings["spherical_aberration"]
ctf_params[:, 7] = image_settings["amplitude_contrast_ratio"]
ctf_params[:, 8] = 0

with open(args.pkl, "wb") as f:
	pickle.dump(ctf_params.astype(np.float32), f)

