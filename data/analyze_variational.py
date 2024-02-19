import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
path = os.path.abspath("model")
sys.path.append(path)
from vae import VAE
from pytorch3d.transforms import matrix_to_axis_angle, rotation_6d_to_matrix



parser_arg = argparse.ArgumentParser()
parser_arg.add_argument("--model_path", type=str, required=True)
args = parser_arg.parse_args()
model_path= args.model_path

model = torch.load(model_path, map_location=torch.device('cpu'))
model.device = "cpu"


all_translatons = model.mean_translation_per_domain
mean_trans = torch.mean(all_translatons, dim=(0, 1))
std_trans = torch.std(all_translatons, dim=(0, 1))

print(mean_trans)
print(std_trans)

all_rotations = matrix_to_axis_angle(rotation_6d_to_matrix(model.mean_rotation_per_domain))
all_angles = all_rotations.norm(p=2, dim=-1, keepdim=True)[:, -1]
all_axis = all_rotations[:, -1]/all_angles
print(all_angles.shape)

all_axis = all_axis.detach().numpy()
dot_prod = np.dot(all_axis, np.array([0, 1, 0]))
plt.hist(dot_prod)
plt.show()
plt.hist(all_angles.detach().numpy().flatten(), bins=50)
plt.show()
plt.scatter(dot_prod, all_angles.detach().numpy(), s=0.1)
plt.show()