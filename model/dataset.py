import os
import torch
import mrcfile
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as tvf
from pytorch3d.transforms import euler_angles_to_matrix

class ImageDataSet(Dataset):
    def __init__(self, apix, side_shape, particles_df, particles_path, down_side_shape=None, down_method="interp"):
        """
        Create a dataset of images and poses
        :param images: torch.tensor(N_images, N_pix_x, N_pix_y) of images
        :param poses: pandas dataframe (N_images, undefined) of information regarding each particles, in Relion star format.
        :param particles_path: str, path to the data folder containing the mrcs files.
        """
        self.side_shape = side_shape
        self.down_method = down_method
        self.apix = apix
        self.particles_path = particles_path
        self.particles_df = particles_df
        euler_angles_degrees = particles_df[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].values
        euler_angles_radians = euler_angles_degrees*np.pi/180
        poses = euler_angles_to_matrix(torch.tensor(euler_angles_radians, dtype=torch.float32), convention="ZYZ")
        poses = torch.transpose(poses, dim0=-2, dim1=-1)
        if "rlnOriginXAngst" in particles_df:
            shiftX = torch.from_numpy(np.array(particles_df["rlnOriginXAngst"], dtype=np.float32))
            shiftY = torch.from_numpy(np.array(particles_df["rlnOriginYAngst"], dtype=np.float32))
        else:
            shiftX = torch.from_numpy(np.array(particles_df["rlnOriginX"] * self.apix, dtype=np.float32))
            shiftY = torch.from_numpy(np.array(particles_df["rlnOriginY"] * self.apix, dtype=np.float32))

        self.poses_translation = torch.tensor(torch.vstack([shiftY, shiftX]).T, dtype=torch.float32)   
        self.poses = poses
        assert self.poses_translation.shape[0] == self.poses.shape[0], "Rotation and translation pose shapes are not matching !"
        #assert torch.max(torch.abs(poses_translation)) == 0, "Only 0 translation supported as poses"
        print("Dataset size:", self.particles_df, "apix:",self.apix)
        print("Normalizing training data")

        self.down_side_shape = side_shape
        if down_side_shape is not None:
            self.down_side_shape = down_side_shape
            self.down_apix = self.side_shape * self.apix /self.down_side_shape

    def standardize(self, images, device="cpu"):
        return (images - self.avg_image.to(device))/self.std_image.to(device)

    def __len__(self):
        return self.particles_df.shape[0]

    def __getitem__(self, idx):
        particles = self.particles_df.iloc[idx]
        try:
            mrc_idx, img_name = particles["rlnImageName"].split("@")
            mrc_idx = int(mrc_idx) - 1
            mrc_path = os.path.join(self.particles_path, img_name)
            with mrcfile.mmap(mrc_path, mode="r", permissive=True) as mrc:
                if mrc.data.ndim > 2:
                    proj = torch.from_numpy(np.array(mrc.data[mrc_idx])).float() #* self.cfg.scale_images
                else:
                    # the mrcs file can contain only one particle
                    proj = torch.from_numpy(np.array(mrc.data)).float() #* self.cfg.scale_images

            # get (1, side_shape, side_shape) proj
            if len(proj.shape) == 2:
                proj = proj[None, :, :]  # add a dummy channel (for consistency w/ img fmt)
            else:
                assert len(proj.shape) == 3 and proj.shape[0] == 1  # some starfile already have a dummy channel

            if self.down_side_shape != self.side_shape:
                if self.down_method == "interp":
                    print("RESIZIIIIIIING")
                    proj = tvf.resize(proj, [self.down_side_shape, ] * 2, antialias=True)
                #elif self.down_method == "fft":
                #    proj = downsample_2d(proj[0, :, :], self.down_side_shape)[None, :, :]
                else:
                    raise NotImplementedError            

            proj = proj[0]

        except Exception as e:
            print(f"WARNING: Particle image {img_name} invalid! Setting to zeros.")
            print(e)
            proj = torch.zeros(self.down_side_shape, self.down_side_shape)

        return idx, proj.flatten(start_dim=-2), self.poses[idx], self.poses_translation[idx]/self.down_apix
