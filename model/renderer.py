import torch
import numpy as np
from time import time
from cryodrgn import mrc
import matplotlib.pyplot as plt


def project(Gauss_mean, Gauss_sigmas, Gauss_amplitudes, grid):
    """
    Project a volumes represented by a GMM into a 2D images, by integrating along the z axis
    Gauss_mean: torch.tensor(batch_size, N_atoms, 3)
    Gauss_sigmas: torch.tensor(N_atoms, 1)
    Gauss_amplitudes: torch.tensor(N_atoms, 1)
    grid: torch.tensor(N_pix**2, 2) where N_pix is the number of pixels on one side of the image
    where N_atoms is the number of atoms in the structure.
    return images: torch.tensor(batch_size, N_pix, N_pix)
    """
    sigmas = 2*Gauss_sigmas**2
    sqrt_amp = torch.sqrt(Gauss_amplitudes)
    #Both proj_x and proj_y are (batch_size, N_atoms, N_pix)
    proj_x = torch.exp(-(Gauss_mean[:, :, None, 0] - grid[None, None, :, 0])**2/sigmas[:, :, None, 0])*sqrt_amp[None, :, :]
    proj_y = torch.exp(-(Gauss_mean[:, :, None, 1] - grid[None, None, :, 1])**2/sigmas[:, :, None, 0])*sqrt_amp[None, :, :]
    images = torch.einsum("b a p1, b a p2 -> b p2 p1", proj_x, proj_y)
    return images

def ctf_corrupt(images, ctf, device):
    """
    Corrupt the images with the ctf
    images: torch.tensor(batch_size, Npix, Npix)
    ctf : torch.tensor(batch_size, N_pix, N_pix)
    return images: torch.tensor(batch_size, N_pix, N_pix) ctf corrupted images.
    """
    fourier_images = torch.fft.fft2(image)
    corrupted_fourier = fourier_images*ctf.to(device)
    corrupted_images = torch.fft.ifft2(corrupted_fourier).real
    return corrupted_images


def structure_to_volume(Gauss_means, Gauss_sigmas, Gauss_amplitudes, grid):
    """
    Project a volumes represented by a GMM into a 2D images, by integrating along the z axis
    Gauss_mean: torch.tensor(batch_size, N_atoms, 3)
    Gauss_sigmas: torch.tensor(N_atoms, 1)
    Gauss_amplitudes: torch.tensor(N_atoms, 1)
    grid: torch.tensor(N_pix**3, 3) where N_pix is the number of pixels on one side of the image
    where N_atoms is the number of atoms in the structure.
    return images: torch.tensor(batch_size, N_pix, N_pix)
    """
    batch_size = Gauss_mean.shape[0]
    N_pix = torch.pow(grid.shape[0], torch.ones(1)*1/3)
    sigmas = 2*Gauss_sigmas**2
    #sqrt_amp = torch.sqrt(Gauss_amplitudes)
    #Both proj_x and proj_y are (batch_size, N_atoms, N_pix)
    denity = torch.sum(torch.sum(-(Gauss_mean[:, :, None] - grid[None, None, :])**2/sigmas[:, :, None, 0], dim=-1)*Gauss_amplitudes[None, :, :], dim=-1)
    density = density.rehsape(batch_size, N_pix, N_pix)
    return images










        



 