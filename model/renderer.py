import torch
import numpy as np
from time import time
import matplotlib.pyplot as plt


def project(Gauss_mean, Gauss_sigmas, Gauss_amplitudes, grid):
    """
    Project a volumes represented by a GMM into a 2D images, by integrating along the z axis
    Gauss_mean: torch.tensor(batch_size, N_atoms, 3)
    Gauss_sigmas: torch.tensor(N_atoms, 1)
    Gauss_amplitudes: torch.tensor(N_atoms, 1)
    grid: grid object
    where N_atoms is the number of atoms in the structure.
    return images: torch.tensor(batch_size, N_pix, N_pix)
    """
    sigmas = 2*Gauss_sigmas**2
    sqrt_amp = torch.sqrt(Gauss_amplitudes)
    #Both proj_x and proj_y are (batch_size, N_atoms, N_pix)
    proj_x = torch.exp(-(Gauss_mean[:, :, None, 0] - grid.line_coords[None, None, :])**2/sigmas[None, :, None,  0])*sqrt_amp[None, :, :]
    proj_y = torch.exp(-(Gauss_mean[:, :, None, 1] - grid.line_coords[None, None, :])**2/sigmas[None, :, None, 0])*sqrt_amp[None, :, :]
    images = torch.einsum("b a p, b a q -> b q p", proj_x, proj_y)
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


def structure_to_volume(Gauss_means, Gauss_sigmas, Gauss_amplitudes, grid, device):
    """
    Project a volumes represented by a GMM into a 2D images, by integrating along the z axis
    Gauss_mean: torch.tensor(batch_size, N_atoms, 3)
    Gauss_sigmas: torch.tensor(N_atoms, 1)
    Gauss_amplitudes: torch.tensor(N_atoms, 1)
    grid: torch.tensor(N_pix,) where N_pix is the number of pixels on one side of the image
    where N_atoms is the number of atoms in the structure.
    return images: torch.tensor(batch_size, N_pix, N_pix)
    """
    batch_size = Gauss_means.shape[0]
    N_pix = torch.pow(grid.line_coords.shape[0], torch.ones(1, device=device)*1/3)
    cubic_root_amp = torch.pow(Gauss_amplitudes, torch.ones(1, device=device)*1/3)
    sigmas = 2*Gauss_sigmas**2
    proj_x = torch.exp(-(Gauss_means[:, :, None, 0] - grid.line_coords[None, None, :])**2/sigmas[None, :, None, 0])*cubic_root_amp[None, :, :]
    proj_y = torch.exp(-(Gauss_means[:, :, None, 1] - grid.line_coords[None, None, :])**2/sigmas[None, :, None, 0])*cubic_root_amp[None, :, :]
    proj_z = torch.exp(-(Gauss_means[:, :, None, 2] - grid.line_coords[None, None, :])**2/sigmas[None, :, None, 0])*cubic_root_amp[None, :, :]
    volumes = torch.einsum("b a p, b a q, b a r -> b p q r", proj_x, proj_y, proj_z)    
    #denity = torch.sum(torch.exp(torch.sum(-(Gauss_means[:, :, None, :] - grid[None, None, :, :])**2/sigmas[None, :, None, :], dim=-1))*Gauss_amplitudes[None, :, None, :], dim=1)
    return volumes
    #density = density.rehsape(batch_size, N_pix, N_pix)
    #return density


def rotate_structure(Gauss_mean, rotation_matrices):
    """
    Rotate a structure to obtain a posed structure.
    Gauss_mean: torch.tensor(batch_size, N_atoms, 3) of atom positions
    rotation_matrices: torch.tensor(batch_size, 3, 3) of rotation_matrices
    return rotated_Gauss_mean: torch.tensor(batch_size, N_atoms, 3)
    """
    rotated_Gauss_mean = torch.einsum("b l k, b a k -> b a l", rotation_matrices, Gauss_mean)
    return rotated_Gauss_mean


def translate_structure(Gauss_mean, translation_vectors):
    """
    Translate a structure to obtain a posed structure.
    Gauss_mean: torch.tensor(batch_size, N_atoms, 3) of atom positions
    translation_vectors: torch.tensor(batch_size, 3) of rotation_matrices
    return translated_Gauss_mean: torch.tensor(batch_size, N_atoms, 3)
    """
    translated_Gauss_mean = Gauss_mean + translation_vectors[:, None, :]
    return translated_Gauss_mean


def get_posed_structure(Gauss_mean, rotation_matrices, translation_vectors):
    """
    Translate a structure to obtain a posed structure.
    Gauss_mean: torch.tensor(batch_size, N_atoms, 3) of atom positions
    rotation_matrices: torch.tensor(batch_size, 3, 3) of rotation_matrices
    translation_vectors: torch.tensor(batch_size, 3) of rotation_matrices
    return posed_Gauss_mean: torch.tensor(batch_size, N_atoms, 3)
    """
    rotated_Gauss_mean = rotate_structure(Gauss_mean, rotation_matrices)
    posed_Gauss_mean = translate_structure(rotated_Gauss_mean, translation_vectors)
    return posed_Gauss_mean


def primal_to_fourier2d(images):
    """
    Computes the fourier transform of the images.
    images: torch.tensor(batch_size, N_pix, N_pix)
    return fourier transform of the images
    """
    r = torch.fft.ifftshift(images, dim=(-2, -1))
    fourier_images = torch.fft.fftshift(torch.fft.fft2(r, dim=(-2, -1), s=(r.shape[-2], r.shape[-1])), dim=(-2, -1))
    return fourier_images

def fourier2d_to_primal(fourier_images):
    """
    Computes the inverse fourier transform
    fourier_images: torch.tensor(batch_size, N_pix, N_pix)
    return fourier transform of the images
    """
    f = torch.fft.ifftshift(fourier_images, dim=(-2, -1))
    r = torch.fft.fftshift(torch.fft.ifft2(f, dim=(-2, -1), s=(f.shape[-2], f.shape[-1])),dim=(-2, -1)).real
    return r


def apply_ctf(images, ctf, indexes):
    """
    apply ctf to images.
    images: torch.tensor(batch_size, N_pix, N_pix)
    ctf: CTF object
    indexes: torch.tensor(batch_size, type=int)
    return ctf corrupted images
    """

    ##### !!!!!!!!!!!!!!!! I AM NOT MULTIPLYING BY -1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    fourier_images = primal_to_fourier2d(images)
    fourier_images *= ctf.compute_ctf(indexes)
    #fourier_images *= -ctf.compute_ctf(indexes)
    ctf_corrupted = fourier2d_to_primal(fourier_images)
    return ctf_corrupted





        



 