import torch
from torch.utils.data import Dataset

class ImageDataSet(Dataset):
    def __init__(self, images_path, poses_path):
        """
        Create a dataset of images and poses
        :param images: torch.tensor(N_images, N_pix_x, N_pix_y) of images
        :param poses: torch.tensor(N_images, 3, 3) of rotation matrices of the pose
        """
        images = torch.load(images_path)
        poses = torch.load(poses_path)
        assert images.shape[0] == poses.shape[0]
        self.images = images
        self.poses = poses

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.poses[idx]