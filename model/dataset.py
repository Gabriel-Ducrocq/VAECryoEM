import torch
from torch.utils.data import Dataset

class ImageDataSet(Dataset):
    def __init__(self, images_path, poses_path, poses_translation_path):
        """
        Create a dataset of images and poses
        :param images: torch.tensor(N_images, N_pix_x, N_pix_y) of images
        :param poses: torch.tensor(N_images, 3, 3) of rotation matrices of the pose
        """
        images = torch.load(images_path)
        poses = torch.load(poses_path)
        poses_translation = torch.load(poses_translation_path)
        assert images.shape[0] == poses.shape[0] and images.shape[0] == poses_translation.shape[0]
        self.images = torch.flatten(images, start_dim=-2, end_dim=-1)
        self.poses = poses
        self.poses_translation = poses_translation

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.poses[idx], self.poses_translation[idx]