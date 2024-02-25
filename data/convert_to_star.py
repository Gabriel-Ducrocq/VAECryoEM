import yaml
import pickle
import argparse
import starfile
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def convert_poses_to_relion(poses_rotations, poses_translations):
	"""
	converts the rotation matrices and translation vectors to a dictionnary
	poses_rotations: np.array(N_images, 3, 3) of rotation matrix in cryoSPHERE conventions
	poses_translations: np.array(N_images, 2) of translation vectors in cryoSPHERE conventions
	return dict
	"""
	assert np.sum(poses_translations) == 0, "Translations must be 0 for now !"
	assert poses_rotations.shape[0] == poses_translations.shape[0], "Different number of rotations and translations"
	## Note that cryoDRGN uses the transposed matrix obtained from the Euler angles.
	## Note also that Relion uses left multiplication with the rotation matrix obtained from the Euler angles (ZYZ). It rotates the grid, so this rotation 
	## matrix should be the transpose of the cryoSphere rotation matrix.
	poses_rotations = poses_rotations.transpose((0, 2, 1))
	poses_rotations = Rotation.from_matrix(poses_rotations)
	poses_rotations_euler = poses_rotations.as_euler("ZYZ", degrees=True)

	poses_dict = {"rlnOriginX":poses_translations[:, 0], "rlnOriginY":poses_translations[:, 1], "rlnAngleRot":poses_rotations_euler[:, 0], 
	"rlnAngleTilt":poses_rotations_euler[:, 1], "rlnAnglePsi":poses_rotations_euler[:, 2]}

	return poses_dict


def convert_ctf_to_relion(particle_mrcs, N_images, N_pixels, apix, ctf_yaml):
	"""
	Converts the ctf as provided by a yaml file to a dictionnary in Relion format.
	particle_mrcs: str, name of the mrc file containing the particles.
	N_images: int, number of particles
	N_pixels: int, number of pixels on one side
	ctf_yaml: dict, containing the parameters of the ctf.
	return a dict tht can be converted in star format
	"""
	ctf_dict = {"rlnOpticsGroupName":"opticsGroup1", "rlnOpticsGroup":1, "rlnMicrographOriginalPixelSize":200, "rlnVoltage":ctf_yaml["accelerating_voltage"], 
			"rlnSphericalAberration":ctf_yaml["spherical_aberration"], "rlnAmplitudeContrast":ctf_yaml["amplitude_contrast_ratio"], "rlnImagePixelSize": apix, 
			"rlnImageSize":N_pixels, "rlnImageDimensionality":2}
	optics_df = pd.DataFrame(ctf_dict, index=[0])

	particle_dict = {"rlnImageName":[f"{i}@{particle_mrcs}" for i in range(1, N_images+1)], "rlnMicrographName":["NoName"]*N_images, "rlnOpticsGroup":[1]*N_images, 
				"rlnDefocusU":[ctf_yaml["dfU"]]*N_images, "rlnDefocusV":[ctf_yaml["dfV"]]*N_images, "rlnDefocusAngle":[ctf_yaml["dfang"]]*N_images}	

	return particle_dict, optics_df


def create_star_file(poses_rotations, poses_translations, particle_mrcs, N_images, N_pixels, apix, ctf_yaml, output_path):
	"""
	create a star file based on the ctf and poses
	poses_rotations: np.array(N_images, 3, 3) of rotation matrix in cryoSPHERE conventions
	poses_tranlations: np.array(N_images, 2) of translation vectors in cryoSPHERE conventions
	particle_mrcs: str, name of the mrc file containing the particles.
	N_images: int, number of particles
	N_pixels: int, number of pixels on one side
	ctf_yaml: dict, containing the parameters of the ctf.
	output_path: str, path to the star file
	"""
	assert output_path.split(".")[-1] == "star", "The file must have a star extension."
	poses_dict = convert_poses_to_relion(poses_rotations, poses_translations)
	particle_dict, optics_df = convert_ctf_to_relion(particle_mrcs, N_images, N_pixels, apix, ctf_yaml)
	particle_dict.update(poses_dict)
	particle_df = pd.DataFrame(particle_dict)
	starfile.write({"optics":optics_df, "particles":particle_df}, output_path)


if __name__ == "__main__":
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

	with open(poses_pkl, "rb") as f:
		poses_rotations, poses_translations = pickle.load(f)

	N_images = experiment_settings["N_images"]
	N_pixels = image_settings["Npix"]
	apix = image_settings["apix"]
	ctf_yaml = image_settings["ctf"]

	poses_dict = create_star_file(poses_rotations, poses_translations, particle_mrcs, N_images, N_pixels, apix, ctf_yaml, output_path)



