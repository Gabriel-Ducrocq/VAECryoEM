import torch
import numpy as np
from torch.utils.data import Dataset
from pytorch3d.transforms import euler_angles_to_matrix

class ImageDataSet(Dataset):
    def __init__(self, images, particles_df):
        """
        Create a dataset of images and poses
        :param images: torch.tensor(N_images, N_pix_x, N_pix_y) of images
        :param poses: pandas dataframe (N_images, undefined) of innformation regarding each particles, in Relion star format.
        """
        euler_angles_degrees = particles_df[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].values
        euler_angles_radians = euler_angles_degrees*np.pi/180
        poses = euler_angles_to_matrix(torch.tensor(euler_angles_radians, dtype=torch.float32), convention="ZYZ")
        poses = torch.transpose(poses, dim0=-2, dim1=-1)
        poses_translation = torch.zeros((particles_df.shape[0], 3), dtype=torch.float32)
        poses_translation[:, :2] = torch.tensor(particles_df[["rlnOriginX", "rlnOriginY"]].values, dtype=torch.float32)
        assert images.shape[0] == poses.shape[0] and images.shape[0] == poses_translation.shape[0]
        assert torch.max(torch.abs(poses_translation)) == 0, "Only 0 translation supported as poses"
        print("Dataset size:", images.shape)
        self.images = torch.flatten(images, start_dim=-2, end_dim=-1)
        self.avg_image = torch.mean(self.images, dim=0, keepdim=True)
        self.std_image = torch.std(self.images, dim=0, keepdim=True)
        print("Normalizing training data")
        self.images = (self.images - self.avg_image)/self.std_image
        self.poses = poses
        self.poses_translation = poses_translation

    def standardize(self, images, device="cpu"):
        return (images - self.avg_image.to(device))/self.std_image.to(device)


    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return idx, self.images[idx], self.poses[idx], self.poses_translation[idx]
