import sys
import os
path = os.path.abspath("model")
sys.path.append(path)
import yaml
import wandb
import torch
import einops
import mrcfile
import warnings
import starfile
import numpy as np
from ctf import CTF
from vae import VAE
from mlp import MLP
import pandas as pd
from tqdm import tqdm
from polymer import Polymer
import torch.nn.functional as F
from dataset import ImageDataSet
from gmm import Gaussian, EMAN2Grid
from Bio.PDB.PDBParser import PDBParser
from biotite.structure.io.pdb import PDBFile
from pytorch3d.transforms import quaternion_to_axis_angle, axis_angle_to_matrix, axis_angle_to_quaternion, quaternion_apply
from pytorch3d.transforms import Transform3d



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

class Mask(torch.nn.Module):

    def __init__(self, im_size, rad, device):
        """
        Mask applied to the image, to exclude parts of the images that are only noise
        im_size: integer, number of pixels a side
        rad: float, radius of the mask
        """
        super(Mask, self).__init__()

        self.device=device
        mask = torch.lt(torch.linspace(-1, 1, im_size)[None]**2 + torch.linspace(-1, 1, im_size)[:, None]**2, rad**2).to(self.device)
        # float for pl ddp broadcast compatible
        self.register_buffer('mask', mask.float())
        self.num_masked = torch.sum(mask).item()

    def forward(self, x):
        """
        Applies the mask to batch of images
        x: torch.tensor(batch_size, im_size, im_size)
        """
        return x * self.mask



def low_pass_images(images, lp_mask2d):
    """
    Low pass filtering of the images.
    images: torch.tensor(batch_size, side_shape, side_shape)
    lp_mask2d: torch.tensor(side_shape, side_shape)
    """
    f_images = primal_to_fourier2d(images)
    f_images = f_images * lp_mask2d
    images = fourier2d_to_primal(f_images).real
    return images


def low_pass_mask2d(shape, apix=1., bandwidth=2):
    freq = np.fft.fftshift(np.fft.fftfreq(shape, apix))
    freq = freq**2
    freq = np.sqrt(freq[:, None] + freq[None, :])

    mask = np.asarray(freq < 1 / bandwidth, dtype=np.float32)
    return mask


def compute_center_of_mass(structure):
    """
    Computes the center of mass of a protein
    :param structure: PDB structure
    :return: np.array(1,3)
    """
    center_mass = np.mean(structure.coord, axis=0)
    return center_mass[None, :]

def center_protein(structure, center_vector):
    """
    Center the protein given ALL its atoms
    :param structure: pdb structure in BioPDB format
    :param center_vector: vector used for the centering of all structures
    :return: PDB structure in BioPDB format, centered structure
    """
    all_coords = np.concatenate([atom.coord[:, None] for atom in structure.get_atoms()], axis=1)
    for index, atom in enumerate(structure.get_atoms()):
        atom.set_coord(all_coords[:, index] - center_vector)

    return structure

def compute_chain_coord(pol, device):
    """
    Get the residue coordinates for each chain
    :param pol: Polymer object
    returns: dictionnary of torch.tensor(N_atom_chains, 3) for each chain
    """
    atom_positions = {}
    N_residues = {}
    chain_ids = sorted(np.unique(pol.chain_id).tolist())
    N_chains = len(chain_ids)
    for n_chain in chain_ids:
        atom_positions[f"chain_{n_chain}"] = torch.tensor(pol.coord[pol.chain_id == n_chain], dtype=torch.float32, device=device)
        N_residues[f"chain_{n_chain}"] = atom_positions[f"chain_{n_chain}"].shape[0]

    return N_chains, atom_positions, N_residues, chain_ids



def parse_yaml(path):
    """
    Parse the yaml file to get the setting for the run
    :param path: str, path to the yaml file
    :return: settings for the run
    """
    with open(path, "r") as file:
        experiment_settings = yaml.safe_load(file)

    with open(experiment_settings["image_yaml"], "r") as file:
        image_settings = yaml.safe_load(file)

    if experiment_settings["device"] == "GPU":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"


    particles_path = experiment_settings["particles_path"]
    apix = image_settings["apix"]
    Npix = image_settings["Npix"]
    Npix_downsize = image_settings["Npix_downsize"]
    apix_downsize = Npix * apix /Npix_downsize
    image_translator = SpatialGridTranslate(D=Npix_downsize, device=device)

    filter_aa = True
    grid = EMAN2Grid(Npix_downsize, apix_downsize, device=device)
    base_structure = Polymer.from_pdb(experiment_settings["base_structure_path"], filter_aa)
    amplitudes = torch.tensor(base_structure.num_electron, dtype=torch.float32, device=device)[:, None]

    N_chains, chain_atom_positions, N_residues, chain_ids = compute_chain_coord(base_structure, device)

    gmm_repr = Gaussian(chain_atom_positions, 
                torch.ones((base_structure.coord.shape[0], 1), dtype=torch.float32, device=device)*image_settings["sigma_gmm"], 
                amplitudes)

    encoder = MLP(Npix_downsize**2,
                  experiment_settings["latent_dimension"] * 2,
                  experiment_settings["encoder"]["hidden_dimensions"], network_type="encoder", device=device,
                  latent_type="continuous")

    total_N_domains = sum(experiment_settings["N_domains"].values())
    decoder = MLP(experiment_settings["latent_dimension"], total_N_domains*6,
                  experiment_settings["decoder"]["hidden_dimensions"], network_type="decoder", device=device)

    if experiment_settings["resume_training"]["model"] == "None":
        vae = VAE(encoder, decoder, device, N_chains=N_chains ,N_domains = experiment_settings["N_domains"], N_residues= N_residues,
                  tau_mask=experiment_settings["tau_mask"], mask_start_values=experiment_settings["mask_start"],
                  latent_type=experiment_settings["latent_type"], latent_dim=experiment_settings["latent_dimension"], chain_ids=chain_ids)
        vae.to(device)
    else:
        vae = torch.load(experiment_settings["resume_training"]["model"])
        vae.to(device)


    if experiment_settings["optimizer"]["name"] == "adam":
        if "learning_rate_mask" not in experiment_settings["optimizer"]:
            optimizer = torch.optim.Adam(vae.parameters(), lr=experiment_settings["optimizer"]["learning_rate"])
            print("LIST PARAMS", (param for param in vae.parameters()))
        else:
            print("Running different LR for the mask")
            list_param = [{"params": param, "lr":experiment_settings["optimizer"]["learning_rate_mask"]} for name, param in
                          vae.named_parameters() if "segment" in name]
            list_param.append({"params": vae.encoder.parameters(), "lr":experiment_settings["optimizer"]["learning_rate"]})
            list_param.append({"params": vae.decoder.parameters(), "lr":experiment_settings["optimizer"]["learning_rate"]})
            optimizer = torch.optim.Adam(list_param)
    else:
        raise Exception("Optimizer must be Adam")


    particles_star = starfile.read(experiment_settings["star_file"])
    ctf_experiment = CTF.from_starfile(experiment_settings["star_file"], apix = apix_downsize, side_shape=Npix_downsize , device=device)
    dataset = ImageDataSet(apix, Npix, particles_star["particles"], particles_path, down_side_shape=Npix_downsize)
    #dataset = ImageDataSet(apix, Npix, particles_star, particles_path, down_side_shape=Npix_downsize)

    scheduler = None
    if "scheduler" in experiment_settings:
        milestones = experiment_settings["scheduler"]["milestones"]
        decay = experiment_settings["scheduler"]["decay"]
        print(f"Using MultiStepLR scheduler with milestones: {milestones} and decay factor {decay}.")
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=decay)

    N_epochs = experiment_settings["N_epochs"]
    batch_size = experiment_settings["batch_size"]
    latent_type = experiment_settings["latent_type"]
    assert latent_type in ["continuous", "categorical"]

    lp_mask2d = low_pass_mask2d(Npix_downsize, apix_downsize, experiment_settings["bandwidth"])
    lp_mask2d = torch.from_numpy(lp_mask2d).to(device).float()

    mask = Mask(Npix_downsize, experiment_settings["mask_radius"], device)

    return vae, image_translator, ctf_experiment, grid, gmm_repr, optimizer, dataset, N_epochs, batch_size, experiment_settings, latent_type, device, \
    scheduler, base_structure, lp_mask2d, mask


class SpatialGridTranslate(torch.nn.Module):

    def __init__(self, D, device=None) -> None:
        super().__init__()
        self.D = D
        # yapf: disable
        coords = torch.stack(torch.meshgrid([
            torch.linspace(-1.0, 1.0, self.D, device=device),
            torch.linspace(-1.0, 1.0, self.D, device=device)],
        indexing="ij"), dim=-1).reshape(-1, 2)
        # yapf: enable
        self.register_buffer("coords", coords)

    def transform(self, images: torch.Tensor, trans: torch.Tensor):
        """
            The `images` are stored in `YX` mode, so the `trans` is also `YX`!

            Supposing that D is 96, a point is at 0.0:
                - adding 48 should move it to the right corner which is 1.0
                    1.0 = 0.0 + 48 / (96 / 2)
                - adding 96(>48) should leave it at 0.0
                    0.0 = 0.0 + 96 / (96 / 2) - 2.0
                - adding -96(<48) should leave it at 0.0
                    0.0 = 0.0 - 96 / (96 / 2) + 2.0

            Input:
                images: (B, NY, NX)
                trans:  (B, T,  2)

            Returns:
                images: (B, T,  NY, NX)
        """
        B, NY, NX = images.shape
        assert self.D == NY == NX
        assert images.shape[0] == trans.shape[0]

        grid = einops.rearrange(self.coords, "N C2 -> 1 1 N C2") - \
            einops.rearrange(trans, "B T C2 -> B T 1 C2") * 2 / self.D
        grid = grid.flip(-1)  # convert the first axis from slow-axis to fast-axis
        grid[grid >= 1] -= 2
        grid[grid <= -1] += 2
        grid.clamp_(-1.0, 1.0)

        sampled = F.grid_sample(einops.rearrange(images, "B NY NX -> B 1 NY NX"), grid, align_corners=True)

        sampled = einops.rearrange(sampled, "B 1 T (NY NX) -> B T NY NX", NX=NX, NY=NY)
        return sampled[:, 0, :, :]




def monitor_training(segments, tracking_metrics, epoch, experiment_settings, vae, optimizer):
    """
    Monitors the training process through wandb and saving masks and models
    :param segments:
    :param tracking_metrics:
    :param epoch:
    :param experiment_settings:
    :param vae:
    :return:
    """
    wandb.log({key: np.mean(val) for key, val in tracking_metrics.items()})
    wandb.log({"epoch": epoch})
    wandb.log({"lr":optimizer.param_groups[0]['lr']})
    N_chains = len(segments)
    chains_translations = {}
    chains = sorted("".join(segments.keys()).split("_"))
    chains.remove("chain")
    for n_chain in chains:
        hard_mask = np.argmax(segments[f"chain_{n_chain}"].detach().cpu().numpy(), axis=-1)
        for l in range(experiment_settings["N_domains"][f"chain_{n_chain}"]):
            wandb.log({f"chain_{n_chain}/segment_{l}": np.sum(hard_mask[0] == l)})

    torch.save(vae, experiment_settings["folder_path"] + "models/full_model" + str(epoch))

def get_atom_positions(residue, name):
    x = residue["CA"].get_coord()
    y = residue["N"].get_coord()
    if name == "GLY":
        z = residue["C"].get_coord()
        return x,y,z

    z = residue["C"].get_coord()
    return x,y,z


def get_backbone(structure):
    N_residue = 0
    residues_indexes = []
    absolute_positions = []
    for model in structure:
        for chain in model:
            for residue in chain:
                residues_indexes.append(N_residue)
                name = residue.get_resname()
                if name not in ["LBV", "NAG", "MAN", "DMS", "BMA"]:
                    x, y, z = get_atom_positions(residue, name)
                    absolute_positions.append(x)
                    absolute_positions.append(y)
                    absolute_positions.append(z)

                    N_residue += 1

    return np.vstack(absolute_positions)



def read_pdb(path):
    """
    Reads a pdb file in a structure object of biopdb
    :param path: str, path to the pdb file.
    :return: a biotite AtomArray or AtomArrayStack
    """
    _, extension = os.path.splitext(path)
    assert extension == "pdb", "The code currently supports only pdb files."
    f = PDBFile.read(path)
    atom_array_stack = f.get_structure()
    if len(atom_array_stack) > 1:
        warnings.warn("More than one structure in the initial pdb file. Using the first one")

    return atom_array_stack[0]


#@torch.jit.script
def compute_rotations_per_residue(quaternions: torch.Tensor, mask: torch.Tensor, device:torch.device):
    """
    Computes the rotation matrix corresponding to each domain for each residue, where the angle of rotation has been
    weighted by the mask value of the corresponding domain.
    :param quaternions: tensor (N_batch, N_domains, 4) of non normalized quaternions defining rotations
    :param mask: tensor (N_batch, N_residues, N_domains)
    :return: tensor (N_batch, N_residues, 3, 3) rotation matrix for each residue
    """
    N_residues = mask.shape[1]
    batch_size = quaternions.shape[0]
    N_domains = mask.shape[-1]
    # NOTE: no need to normalize the quaternions, quaternion_to_axis does it already.
    rotation_per_domains_axis_angle = quaternion_to_axis_angle(quaternions)
    mask_rotation_per_domains_axis_angle = mask[:, :, :, None] * rotation_per_domains_axis_angle[:, None, :, :]

    mask_rotation_matrix_per_domain_per_residue = axis_angle_to_matrix(mask_rotation_per_domains_axis_angle)
    # Transposed here because pytorch3d has right matrix multiplication convention.
    # mask_rotation_matrix_per_domain_per_residue = torch.transpose(mask_rotation_matrix_per_domain_per_residue, dim0=-2, dim1=-1)
    overall_rotation_matrices = torch.zeros((batch_size, N_residues, 3, 3), device=device)
    overall_rotation_matrices[:, :, 0, 0] = 1
    overall_rotation_matrices[:, :, 1, 1] = 1
    overall_rotation_matrices[:, :, 2, 2] = 1
    for i in range(N_domains):
        overall_rotation_matrices = torch.matmul(mask_rotation_matrix_per_domain_per_residue[:, :, i, :, :],
                                                 overall_rotation_matrices)

    return overall_rotation_matrices

def compute_rotations_per_residue_einops(quaternions, mask, device):
    """
    Computes the rotation matrix corresponding to each domain for each residue, where the angle of rotation has been
    weighted by the mask value of the corresponding domain.
    :param quaternions: tensor (N_batch, N_domains, 4) of non normalized quaternions defining rotations
    :param mask: tensor (N_batch, N_residues, N_domains)
    :return: tensor (N_batch, N_residues, 3, 3) rotation matrix for each residue
    """

    N_residues = mask.shape[1]
    batch_size = quaternions.shape[0]
    N_domains = mask.shape[-1]
    # NOTE: no need to normalize the quaternions, quaternion_to_axis does it already.
    rotation_per_domains_axis_angle = quaternion_to_axis_angle(quaternions)
    mask_rotation_per_domains_axis_angle = mask[:, :, :, None] * rotation_per_domains_axis_angle[:, None, :, :]
    mask_rotation_matrix_per_domain_per_residue = axis_angle_to_matrix(mask_rotation_per_domains_axis_angle)
    ## Flipping to keep in line with the previous implementation
    mask_rotation_matrix_per_domain_per_residue = torch.einsum("brdle->dbrle", mask_rotation_matrix_per_domain_per_residue).flip(0)
    dimensions = ",".join([f"b r a{i} a{i+1}" for i in range(N_domains)])
    dimensions += f"-> b r a0 a{N_domains}"
    overall_rotation_matrices = einops.einsum(*mask_rotation_matrix_per_domain_per_residue, dimensions)
    return overall_rotation_matrices


def rotate_residues_einops(atom_positions, quaternions, segments, device):
    """
    Computes the rotation matrix corresponding to each domain for each residue, where the angle of rotation has been
    weighted by the mask value of the corresponding domain.
    :param positions: dictionnary of atom positions per chain torch.tensor(N_residues_chains, 3)
    :param quaternions: tensor (N_batch, N_domains, 4) of non normalized quaternions defining rotations
    :param mask: tensor (N_batch, N_residues, N_domains)
    :return: tensor (N_batch, N_residues, 3, 3) rotation matrix for each residue
    """
    chain_atom_positions = {}
    chains = sorted("".join(atom_positions.keys()).split("_"))
    chains.remove("chain")
    for n_chains in chains:
        N_residues = segments[f"chain_{n_chains}"].shape[1]
        batch_size = quaternions[f"chain_{n_chains}"].shape[0]
        N_domains = segments[f"chain_{n_chains}"].shape[-1]
        # NOTE: no need to normalize the quaternions, quaternion_to_axis does it already.
        rotation_per_domains_axis_angle = quaternion_to_axis_angle(quaternions[f"chain_{n_chains}"])
        #The below tensor is [N_batch, N_residues, N_domains, 3]
        segments_rotation_per_domains_axis_angle = segments[f"chain_{n_chains}"][:, :, :, None] * rotation_per_domains_axis_angle[:, None, :, :]
        segments_rotation_per_domains_quaternions = axis_angle_to_quaternion(segments_rotation_per_domains_axis_angle)
        print(segments_rotation_per_domains_quaternions[:, :, 0, :].shape)
        new_atom_positions = quaternion_apply(segments_rotation_per_domains_quaternions[:, :, 0, :], atom_positions[f"chain_{n_chains}"])
        for dom in range(1, N_domains):
            new_atom_positions = quaternion_apply(segments_rotation_per_domains_quaternions[:, :, dom, :], new_atom_positions)

        chain_atom_positions[f"chain_{n_chains}"] = new_atom_positions

    return chain_atom_positions

    # NOTE: no need to normalize the quaternions, quaternion_to_axis does it already.
    #rotation_per_domains_axis_angle = quaternion_to_axis_angle(quaternions)
    #The below tensor is [N_batch, N_residues, N_domains, 3]
    #mask_rotation_per_domains_axis_angle = mask[:, :, :, None] * rotation_per_domains_axis_angle[:, None, :, :]
    #mask_rotation_per_domains_quaternions = axis_angle_to_quaternion(mask_rotation_per_domains_axis_angle)
    #T = Transform3d(dtype=torch.float32, device = device)
    #atom_positions = quaternion_apply(mask_rotation_per_domains_quaternions[:, :, 0, :], atom_positions)
    #for dom in range(1, N_domains):
    #    atom_positions = quaternion_apply(mask_rotation_per_domains_quaternions[:, :, dom, :], atom_positions)

    #return atom_positions



def compute_translations_per_residue(translation_vectors, segments):
    """
    Computes one translation vector per residue based on the mask
    :param translation_vectors: dictionnary of torch.tensor (Batch_size, N_domains_chain, 3) translations for each domain
    :param segments: translation_vectors: dictionnary of torch.tensor(N_batch, N_residues_chains, N_domains_chains) weights of the attention segments
    :return: translation_vectors: dictionnary of translation per residue per chain torch.tensor(batch_size, N_residues_chains, 3)
    """
    #### How is it possible to compute this product given the two tensor size
    N_chains = len(segments)
    chains_translations = {}
    chains = sorted("_".join(segments.keys()).split("_"))
    chains = list(filter(lambda x: x != "chain", chains))
    print(chains)
    for n_chain in chains:
        translation_per_residue = torch.einsum("bij, bjk -> bik", segments[f"chain_{n_chain}"], translation_vectors[f"chain_{n_chain}"])
        chains_translations[f"chain_{n_chain}"] = translation_per_residue

    return chains_translations


def deform_structure_bis(atom_positions, translation_per_residue, quaternions, segments, device):
    """
    Note that the reference frame absolutely needs to be the SAME for all the residues (placed in the same spot),
     otherwise the rotation will NOT be approximately rigid !!!
    :param positions: dictionnary of positions of atom for each chain: torch.tensor(N_residues_chain, 3)
    :param translation_vectors: dictionnary of translations vectors per chain: tensor (Batch_size, N_residues_chains, 3)
    :param rotations_per_residue: dictionnary of rotation matrices per domain for each chain: tensor (N_batch, N_domains_chains, 3, 3)
    :return: tensor (Batch_size, 3*N_residues, 3) corresponding to translated structure, tensor (3*N_residues, 3)
            of translation vectors
    """
    ## We displace the structure, using an interleave because there are 3 consecutive atoms belonging to one
    ## residue.
    ##We compute the rotated residues, where this axis of rotation is at the origin.
    transformed_atom_positions = rotate_residues_einops(atom_positions, quaternions, segments, device)
    new_atom_positions = []
    N_chains = len(segments)
    chains_translations = {}
    chains = sorted("".join(segments.keys()).split("_"))
    chains.remove("chain")
    for n_chain in chains:
        chain_new_atom_positions = transformed_atom_positions[f"chain_{n_chain}"]  + translation_per_residue[f"chain_{n_chain}"]
        new_atom_positions.append(chain_new_atom_positions) 

    new_atom_positions = torch.concatenate(new_atom_positions, dim=1)
    return new_atom_positions



def deform_structure(atom_positions, translation_per_residue, rotations_per_residue):
    """
    Note that the reference frame absolutely needs to be the SAME for all the residues (placed in the same spot),
     otherwise the rotation will NOT be approximately rigid !!!
    :param positions: torch.tensor(N_residues, 3)
    :param translation_vectors: translations vectors:
            tensor (Batch_size, N_residues, 3)
    :param rotations_per_residue: tensor (N_batch, N_residues, 3, 3) of rotation matrices per residue
    :return: tensor (Batch_size, 3*N_residues, 3) corresponding to translated structure, tensor (3*N_residues, 3)
            of translation vectors
    """
    ## We displace the structure, using an interleave because there are 3 consecutive atoms belonging to one
    ## residue.
    ##We compute the rotated residues, where this axis of rotation is at the origin.
    transformed_atom_positions = torch.einsum("lbjk, bk->lbj", rotations_per_residue, atom_positions)
    #transformed_atom_positions = torch.matmul(rotation_per_atom, atom_positions[None, :, :, None])
    new_atom_positions = transformed_atom_positions + translation_per_residue
    return new_atom_positions



