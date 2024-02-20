import yaml
import argparse
import starfile
import pandas as pd


parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--mrcs_file', type=str, required=True)
parser_arg.add_argument('--images_yaml', type=str, required=True)
parser_arg.add_argument('--o', type=str, required=True)
parser_arg.add_argument('--parameters_yaml', type=str, required=True)
args = parser_arg.parse_args()
particle_mrcs = args.mrcs_file
images_yaml = args.images_yaml
output_path = args.o
parameters_yaml = args.parameters_yaml

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

particle_df = pd.DataFrame(particle_dict)

starfile.write({"optics":optics_df, "particles":particle_df}, output_path)


