import torch
import numpy as np
from model.dataset import ImageDataSet
from torch.utils.data import DataLoader
from pytorch3d.transforms import quaternion_to_axis_angle


folder_path = "../data/debug_run2/"
model_path = "../data/debug_run2/models/full_model4597"
model = torch.load(model_path, map_location=torch.device('cpu'))
model.device="cpu"
dataset = ImageDataSet("../data/debug_run2/ImageDataSet", "../data/debug_run2/poses")
data_loader = iter(DataLoader(dataset, batch_size=100, shuffle=False))
all_rotation_per_domain = []
all_translation_per_domain = []
all_latent_means = []
for i, (batch_images, batch_poses) in enumerate(data_loader):
    print(i)
    latent_variable, latent_mean, latent_std = model.sample_latent(batch_images)
    quaternions_per_domain, translation_per_domain = model.decode(latent_mean)
    axis_angle_per_domain = quaternion_to_axis_angle(quaternions_per_domain)
    all_rotation_per_domain.append(axis_angle_per_domain)
    all_translation_per_domain.append(translation_per_domain)
    all_latent_means.append(latent_mean)

all_rotation_per_domain = torch.concat(all_rotation_per_domain, dim=0).detach().numpy()
all_translation_per_domain = torch.concat(all_translation_per_domain, dim=0).detach().numpy()
all_latent_means = torch.concat(all_latent_means, dim=0).detach().numpy()


np.save(folder_path + "latent_distrib.npy", all_latent_means)
np.save(folder_path + "all_rotations.npy", all_rotation_per_domain)
np.save(folder_path + "all_translations", all_translation_per_domain)










