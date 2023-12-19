import torch
import numpy as np
import argparse
from pytorch3d.transforms import matrix_to_euler_angles

parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--poses_torch', type=str, required=True)
parser_arg.add_argument('--o', type=str, required=True)

args = parser_arg.parse_args()
input_path = args.poses_torch
output_path = args.o

rotations = torch.load(input_path)
rotations_euler = matrix_to_euler_angles(rotations, convention="ZYZ").detach().cpu().numpy()

with open(output_path, "w") as f:
	f.write("data_\n")
	f.write("loop_\n")
	f.write("_rlnAngleRot\n")
	f.write("_rlnAngleTilt\n")
	f.write("_rlnAnglePsi\n")
	#Must convert to degrees to gets the Relion .star format for the poses
	rotations_euler *= 180/np.pi
	for i in range(rotations_euler.shape[0]):
		rot_eul = rotations_euler[i]
		f.write(f"{rot_eul[0]} {rot_eul[1]} {rot_eul[2]}\n")




