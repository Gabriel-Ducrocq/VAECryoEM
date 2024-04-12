import mrc
import time
import argparse
import numpy as np
from EMAN2 import *
from tqdm import tqdm
from convert_to_star import create_star_file
from scipy.spatial.transform import Rotation

parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--volume_path', type=str, required=True)
parser_arg.add_argument('--folder_experiment', type=str, required=True)
args = parser_arg.parse_args()
volume_path = args.volume_path
folder_experiment = args.folder_experiment
Nimages = args.Nimages

with open(f"{folder_experiment}/parameters.yaml", "r") as file:
    experiment_settings = yaml.safe_load(file)

with open(f"{folder_experiment}/images.yaml", "r") as file:
    image_settings = yaml.safe_load(file)

N_images = experiment_settings["N_images"]
apix = image_settings["apix"]
Npix = image_settings["Npix"]


vol = EMData(volume_path)
all_rotations = Rotation.random(num=Nimages)
all_rotations_eman2 = all_rotations.as_euler("ZXZ", degrees=True)
all_rotations_relion = all_rotations.as_euler("ZYZ", degrees=True)

start = time.time()
all_images = []
for rot in tqdm(all_rotations_eman2):
	t = Transform({"type":"eman","az":rot[0],"alt":rot[1], "phi":rot[2]})
	proj = vol.project("standard", t)
	image = np.flip(proj.numpy(), axis=0)
	all_images.append(image)

all_images = np.stack(all_images, axis=0)
mrc.MRCFile.write(f"{folder_experiment}particles.mrcs", all_images, Apix=apix, is_vol=False)
output_path = f"{folder_experiment}particles.star"
create_star_file(all_rotations_eman2, np.zeros((Nimages, 2)), "particles.mrcs", N_images, Npix, apix, image_settings["ctf"], output_path)
end = time.time()
print(end-start)