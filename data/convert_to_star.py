import yaml
import pickle
import argparse
import starfile
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--mrcs_file', type=str, required=True)
parser_arg.add_argument('--images_yaml', type=str, required=True)
parser_arg.add_argument('--o', type=str, required=True)
parser_arg.add_argument('--parameters_yaml', type=str, required=True)
parser_arg.add_argument("--poses_pkl", type=str, required=False)
args = parser_arg.parse_args()
particle_mrcs = args.mrcs_file
images_yaml = args.images_yaml
output_path = args.o
parameters_yaml = args.parameters_yaml
poses_pkl = args.poses_pkl

with open(parameters_yaml, "r") as file:
    experiment_settings = yaml.safe_load(file)

with open(images_yaml, "r") as file:
    image_settings = yaml.safe_load(file)




N_images = experiment_settings["N_images"]
N_pixels = image_settings["N_pixels_per_axis"][0]
Apix = (image_settings["image_upper_bounds"][0] - image_settings["image_lower_bounds"][0])/N_pixels
ctf_yaml = image_settings["renderer"]
ctf_dict = {"rlnOpticsGroupName":"opticsGroup1", "rlnOpticsGroup":1, "rlnMicrographOriginalPixelSize":200, "rlnVoltage":ctf_yaml["accelerating_voltage"], 
			"rlnSphericalAberration":ctf_yaml["spherical_aberration"], "rlnAmplitudeContrast":ctf_yaml["amplitude_contrast_ratio"], "rlnImagePixelSize": Apix, 
			"rlnImageSize":N_pixels, "rlnImageDimensionality":2}
optics_df = pd.DataFrame(ctf_dict, index=[0])

particle_dict = {"rlnImageName":[f"{i}@{particle_mrcs}" for i in range(1, N_images+1)], "rlnMicrographName":["NoName"]*N_images, "rlnOpticsGroup":[1]*N_images, 
				"rlnDefocusU":[ctf_yaml["dfU"]]*N_images, "rlnDefocusV":[ctf_yaml["dfV"]]*N_images, "rlnDefocusAngle":[ctf_yaml["dfang"]]*N_images}


if poses_pkl:
	with open(poses_pkl, "rb") as f:
		poses_rotations, poses_translation = pickle.load(f)

	assert np.sum(poses_translation) == 0, "Translations must be 0 for now !"
	poses_rotations = poses_rotations.transpose((0, 2, 1))
	poses_rotations = Rotation.from_matrix(poses_rotations)
	poses_rotations_euler = poses_rotations.as_euler("ZYZ", degrees=True)

	poses_dict = {"_rlnOriginX":poses_translation[:, 0], "_rlnOriginY":poses_translation[:, 1], "_rlnAngleRot":poses_rotations_euler[:, 0], 
	"_rlnAngleTilt":poses_rotations_euler[:, 1], "_rlnAnglePsi":poses_rotations_euler[:, 2]}

	particle_dict.update(poses_dict)

particle_df = pd.DataFrame(particle_dict)

starfile.write({"optics":optics_df, "particles":particle_df}, output_path)


