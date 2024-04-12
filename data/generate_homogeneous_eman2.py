import time
import argparse
from EMAN2 import *
from tqdm import tqdm
from scipy.spatial.transform import Rotation

parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--volume_path', type=str, required=True)
parser_arg.add_argument('--Nimages', type=int, required=True)
args = parser_arg.parse_args()
volume_path = args.volume_path
Nimages = args.Nimages


volume = EMData(volume_path)
all_rotations = Rotation.random(num=Nimages)
all_rotations_eman2 = all_rotations.as_euler("ZXZ", degrees=True)
all_rotations_relion = all_rotations.as_euler("ZYZ", degrees=True)

start = time.time()
for rot in tqdm(all_rotations_eman2):
	t = Transform({"type":"eman","az":rot[0],"alt":rot[1], "phi":rot[2]})
	proj = vol.project("standard", t)

end = time.time()
print(end-start)