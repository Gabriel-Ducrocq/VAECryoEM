import torch
from torch.utils.data import Dataset
from pytorch3d.transforms import euler_angles_to_matrix

class ImageDataSet(Dataset):
    def __init__(self, images, particles_df):
        """
        Create a dataset of images and poses
        :param images: torch.tensor(N_images, N_pix_x, N_pix_y) of images
        :param poses: pandas dataframe (N_images, undefined) of innformation regarding each particles, in Relion star format.
        """
        poses = euler_angles_to_matrix(torch.tensor(particles_df[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].values, dtype=torch.float32), convention="ZYZ")
        poses = torch.transpose(poses, dim0=-2, dim1=-1)
        poses_translation = torch.zeros((particles_df.shape[0], 3), dtype=torch.float32)
        poses_translation[:, :2] = torch.tensor(particles_df[["rlnOriginX", "rlnOriginY"]].values, dtype=torch.float32)
        assert images.shape[0] == poses.shape[0] and images.shape[0] == poses_translation.shape[0]
        assert torch.max(torch.abs(poses_translation)) == 0, "Only 0 translation supported as poses"
        self.images = torch.flatten(images, start_dim=-2, end_dim=-1)
        self.poses = poses
        self.poses_translation = poses_translation

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return idx, self.images[idx], self.poses[idx], self.poses_translation[idx]
