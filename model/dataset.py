import os
import torch
import mrcfile
import numpy as np
from time import time
from torch.utils.data import Dataset
import torchvision.transforms.functional as tvf
from pytorch3d.transforms import euler_angles_to_matrix



class Mask(torch.nn.Module):

    def __init__(self, im_size, rad):
        super(Mask, self).__init__()

        mask = torch.lt(torch.linspace(-1, 1, im_size)[None]**2 + torch.linspace(-1, 1, im_size)[:, None]**2, rad**2)
        # float for pl ddp broadcast compatible
        self.register_buffer('mask', mask.float())
        self.num_masked = torch.sum(mask).item()

    def forward(self, x):
        return x * self.mask



class ImageDataSet(Dataset):
    def __init__(self, apix, side_shape, particles_df, particles_path, down_side_shape=None, down_method="interp", invert_data=True, rad_mask=None):
        """
        #Create a dataset of images and poses
        #:param apix: float, size of a pixel in Å.
        #:param side_shape: integer, number of pixels on each side of a picture. So the picture is a side_shape x side_shape array
        #:param particle_df: particles dataframe coming from a star file
        #:particles_path: string, path to the folder containing the mrcs files. It is appended to the path present in the star file.
        #:param down_side_shape: integer, number of pixels of the downsampled images. If no downampling, set down_side_shape = side_shape. 
        """

        self.side_shape = side_shape
        self.down_method = down_method
        self.apix = apix
        self.particles_path = particles_path
        self.particles_df = particles_df
        self.mask = None
        if rad_mask is not None:
            self.mask = Mask(side_shape, rad_mask)

        print(particles_df.columns)
        #Reading the euler angles and turning them into rotation matrices. 
        euler_angles_degrees = particles_df[["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].values
        euler_angles_radians = euler_angles_degrees*np.pi/180
        poses = euler_angles_to_matrix(torch.tensor(euler_angles_radians, dtype=torch.float32), convention="ZYZ")
        #Transposing because ReLion has a clockwise convention, while we use a counter-clockwise convention.
        poses = torch.transpose(poses, dim0=-2, dim1=-1)

        #Reading the translations. ReLion may express the translations divided by apix. So we need to multiply by apix to recover them in Å
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
        print("Dataset size:", self.particles_df.shape[0], "apix:",self.apix)
        print("Normalizing training data")

        #If a downsampling is wanted, recompute the new apix and set the new down_side_shape
        self.down_side_shape = side_shape
        if down_side_shape is not None:
            self.down_side_shape = down_side_shape
            self.down_apix = self.side_shape * self.apix /self.down_side_shape

        self.invert_data = invert_data
        self.f_std = None
        self.f_mu = None
        self.estimate_normalization()

    def estimate_normalization(self):
        if self.f_mu is None and self.f_std is None:
            f_sub_data = []
            # I have checked that the standard deviation of 10/100/1000 particles is similar
            for i in range(0, len(self), len(self) // 100):
                _, _, _, _, fproj = self[i]
                f_sub_data.append(fproj)

            f_sub_data = torch.cat(f_sub_data, dim=0)
            self.f_mu = 0.0  # just follow cryodrgn
            self.f_std = torch.std(f_sub_data).item()
            print("Estimated std", self.f_std)
        else:
            raise Exception("The normalization factor has been estimated!")

    def standardize(self, images, device="cpu"):
        return (images - self.avg_image.to(device))/self.std_image.to(device)

    def __len__(self):
        return self.particles_df.shape[0]

    def __getitem__(self, idx):
        """
        #Return a batch of true images, as 2d array !
        # return: the set of indexes queried for the batch, the corresponding images as a torch.tensor((batch_size, side_shape, side_shape)), 
        # the corresponding poses rotation matrices as torch.tensor((batch_size, 3, 3)), the corresponding poses translations as torch.tensor((batch_size, 2))
        # NOTA BENE: the convention for the rotation matrix is left multiplication of the coordinates of the atoms of the protein !!
        """
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
                    proj = tvf.resize(proj, [self.down_side_shape, ] * 2, antialias=True)
                #elif self.down_method == "fft":
                #    proj = downsample_2d(proj[0, :, :], self.down_side_shape)[None, :, :]
                else:
                    raise NotImplementedError            

            proj = proj[0]
            if self.mask is not None:
                proj = self.mask(proj)

            fproj = primal_to_fourier_2d(proj)
            if self.f_mu is not None:
                fproj = (fproj - self.f_mu) / self.f_std
                proj = fourier_to_primal_2d(fproj).real

        except Exception as e:
            mrc_idx, img_name = particles["rlnImageName"].split("@")
            print(os.path.join(self.particles_path, img_name))
            print(f"WARNING: Particle image {img_name} invalid! Setting to zeros.")
            print(e)
            proj = torch.zeros(self.down_side_shape, self.down_side_shape)

        #if self.invert_data:
        #    print("INVERTING")
        #    proj *= -1

        return idx, proj, self.poses[idx], self.poses_translation[idx]/self.down_apix, fproj



def primal_to_fourier_2d(images):
    """
    Computes the fourier transform of the images.
    images: torch.tensor(batch_size, N_pix, N_pix)
    return fourier transform of the images
    """
    r = torch.fft.ifftshift(images, dim=(-2, -1))
    fourier_images = torch.fft.fftshift(torch.fft.fft2(r, dim=(-2, -1), s=(r.shape[-2], r.shape[-1])), dim=(-2, -1))
    return fourier_images

def fourier_to_primal_2d(fourier_images):
    """
    Computes the inverse fourier transform
    fourier_images: torch.tensor(batch_size, N_pix, N_pix)
    return fourier transform of the images
    """
    f = torch.fft.ifftshift(fourier_images, dim=(-2, -1))
    r = torch.fft.fftshift(torch.fft.ifft2(f, dim=(-2, -1), s=(f.shape[-2], f.shape[-1])),dim=(-2, -1)).real
    return r



class StarfileDataSet(Dataset):

    #def __init__(self, cfg: StarfileDatasetConfig):
    def __init__(self, apix, side_shape, particles_df, particles_path, down_side_shape=None, down_method="interp", invert_data=True):
        super().__init__()
        self.cfg = cfg
        starfile_path = Path(cfg.starfile_path).resolve()
        if cfg.dataset_dir is None:
            cfg.dataset_dir = Path(starfile_path).resolve().parent
        else:
            cfg.dataset_dir = Path(cfg.dataset_dir)

        self.df = starfile.read(Path(cfg.starfile_path))

        if "optics" in self.df:
            optics_df = self.df["optics"]
            particles_df = self.df["particles"]
        else:
            optics_df = None
            particles_df = self.df
        self.particles_df = particles_df

        self.apix = apix

        if cfg.side_shape is None:
            tmp_mrc_path = osp.join(cfg.dataset_dir, particles_df["rlnImageName"][0].split('@')[-1])
            with mrcfile.mmap(tmp_mrc_path, mode="r", permissive=True) as m:
                self.side_shape = m.data.shape[-1]
            print(f"Infer dataset side_shape={self.side_shape} from the 1st particle.")
        else:
            self.side_shape = cfg.side_shape

        self.num_proj = len(particles_df)

        self.down_side_shape = self.side_shape
        self.down_apix = self.apix
        if cfg.down_side_shape is not None:
            self.down_side_shape = cfg.down_side_shape
            self.down_apix = self.side_shape * self.apix / cfg.down_side_shape

        if cfg.mask_rad is not None:
            self.mask = Mask(self.down_side_shape, cfg.mask_rad)

        self.f_mu = None
        self.f_std = None

    def __len__(self):
        return self.num_proj

    def estimate_normalization(self):
        if self.f_mu is None and self.f_std is None:
            f_sub_data = []
            # I have checked that the standard deviation of 10/100/1000 particles is similar
            for i in range(0, len(self), len(self) // 100):
                f_sub_data.append(self[i]["fproj"])
            f_sub_data = torch.cat(f_sub_data, dim=0)
            # self.f_mu = torch.mean(f_sub_data)
            self.f_mu = 0.0  # just follow cryodrgn
            self.f_std = torch.std(f_sub_data).item()
        else:
            raise Exception("The normalization factor has been estimated! Std:", self.f_std)

    def __getitem__(self, idx):
        item_row = self.particles_df.iloc[idx]
        try:
            img_name_raw = item_row["rlnImageName"]
            in_mrc_idx, img_name = item_row["rlnImageName"].split("@")
            in_mrc_idx = int(in_mrc_idx) - 1
            print("MRC cryostar", in_mrc_idx, img_name)
            mrc_path = osp.join(self.cfg.dataset_dir, img_name)
            with mrcfile.mmap(mrc_path, mode="r", permissive=True) as mrc:
                if mrc.data.ndim > 2:
                    proj = torch.from_numpy(np.array(mrc.data[in_mrc_idx])).float() * self.cfg.scale_images
                else:
                    # the mrcs file can contain only one particle
                    proj = torch.from_numpy(np.array(mrc.data)).float() * self.cfg.scale_images

            # get (1, side_shape, side_shape) proj
            if len(proj.shape) == 2:
                proj = proj[None, :, :]  # add a dummy channel (for consistency w/ img fmt)
            else:
                assert len(proj.shape) == 3 and proj.shape[0] == 1  # some starfile already have a dummy channel

            # down-sample
            if self.down_side_shape != self.side_shape:
                if self.cfg.down_method == "interp":
                    proj = tvf.resize(proj, [self.down_side_shape, ] * 2, antialias=True)
                elif self.cfg.down_method == "fft":
                    proj = downsample_2d(proj[0, :, :], self.down_side_shape)[None, :, :]
                else:
                    raise NotImplementedError

            if self.cfg.mask_rad is not None:
                proj = self.mask(proj)

        except Exception as e:
            print(f"WARNING: Particle image {img_name_raw} invalid! Setting to zeros.")
            print(e)
            proj = torch.zeros(1, self.down_side_shape, self.down_side_shape)

        if self.cfg.power_images != 1.0:
            proj *= self.cfg.power_images

        # Generate CTF from CTF paramaters
        defocusU = torch.from_numpy(np.array(item_row["rlnDefocusU"] / 1e4, ndmin=2)).float()
        defocusV = torch.from_numpy(np.array(item_row["rlnDefocusV"] / 1e4, ndmin=2)).float()
        angleAstigmatism = torch.from_numpy(np.radians(np.array(item_row["rlnDefocusAngle"], ndmin=2))).float()

        # Read "GT" orientations
        if self.cfg.ignore_rots:
            rotmat = torch.eye(3).float()
        else:
            # yapf: disable
            rotmat = torch.from_numpy(euler_angles2matrix(
                np.radians(-item_row["rlnAngleRot"]),
                # np.radians(particle["rlnAngleTilt"]) * (-1 if self.cfg.invert_hand else 1),
                np.radians(-item_row["rlnAngleTilt"]),
                np.radians(-item_row["rlnAnglePsi"]))
            ).float()
            # yapf: enable

        # Read "GT" shifts
        if self.cfg.ignore_trans:
            shiftX = torch.tensor([0.])
            shiftY = torch.tensor([0.])
        else:
            # support early starfile formats
            # Particle translations used to be in pixels (rlnOriginX and rlnOriginY) but this changed to Angstroms
            # (rlnOriginXAngstrom and rlnOriginYAngstrom) in relion 3.1.
            # https://relion.readthedocs.io/en/release-3.1/Reference/Conventions.html
            if "rlnOriginXAngst" in item_row:
                shiftX = torch.from_numpy(np.array(item_row["rlnOriginXAngst"], dtype=np.float32))
                shiftY = torch.from_numpy(np.array(item_row["rlnOriginYAngst"], dtype=np.float32))
            else:
                shiftX = torch.from_numpy(np.array(item_row["rlnOriginX"] * self.apix, dtype=np.float32))
                shiftY = torch.from_numpy(np.array(item_row["rlnOriginY"] * self.apix, dtype=np.float32))

        fproj = primal_to_fourier_2d(proj)

        if self.f_mu is not None:
            fproj = (fproj - self.f_mu) / self.f_std
            proj = fourier_to_primal_2d(fproj).real

        in_dict = {
            "proj": proj,
            "rotmat": rotmat,
            "defocusU": defocusU,
            "defocusV": defocusV,
            "shiftX": shiftX,
            "shiftY": shiftY,
            "angleAstigmatism": angleAstigmatism,
            "idx": torch.tensor(idx, dtype=torch.long),
            "fproj": fproj,
            "imgname_raw": img_name_raw
        }

        if "rlnClassNumber" in item_row:
            in_dict["class_id"] = item_row["rlnClassNumber"]

        return in_dict
