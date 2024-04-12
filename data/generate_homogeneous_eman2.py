import time
import argparse
import numpy as np
from EMAN2 import *
from tqdm import tqdm
from scipy.spatial.transform import Rotation

parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--volume_path', type=str, required=True)
parser_arg.add_argument('--Nimages', type=int, required=True)
args = parser_arg.parse_args()
volume_path = args.volume_path
Nimages = args.Nimages


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
end = time.time()
print(end-start)