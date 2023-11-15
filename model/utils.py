import sys
import os
path = os.path.abspath("model")
sys.path.append(path)
import yaml
import wandb
import torch
import numpy as np
from vae import VAE
from mlp import MLP
from renderer import Renderer
from dataset import ImageDataSet
from Bio.PDB.PDBParser import PDBParser
from pytorch3d.transforms import quaternion_to_axis_angle, axis_angle_to_matrix


def compute_center_of_mass(structure):
    """
    Computes the center of mass of a protein
    :param structure: PDB structure
    :return: np.array(1,3)
    """
    all_coords = np.concatenate([atom.coord[:, None] for atom in structure.get_atoms()], axis=1)
    center_mass = np.mean(all_coords, axis=1)
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

    for mask_prior_key in experiment_settings["mask_prior"].keys():
        experiment_settings["mask_prior"][mask_prior_key]["mean"] = torch.tensor(experiment_settings["mask_prior"][mask_prior_key]["mean"],
                                                                                 dtype=torch.float32, device=device)
        experiment_settings["mask_prior"][mask_prior_key]["std"] = torch.tensor(experiment_settings["mask_prior"][mask_prior_key]["std"],
                                                                                 dtype=torch.float32, device=device)

    if experiment_settings["latent_type"] == "continuous":
        encoder = MLP(image_settings["N_pixels_per_axis"][0] * image_settings["N_pixels_per_axis"][1],
                      experiment_settings["latent_dimension"] * 2,
                      experiment_settings["encoder"]["hidden_dimensions"], network_type="encoder", device=device,
                      latent_type="continuous")
        decoder = MLP(experiment_settings["latent_dimension"], experiment_settings["N_domains"]*6,
                      experiment_settings["decoder"]["hidden_dimensions"], network_type="decoder", device=device)
    else:
        encoder = MLP(image_settings["N_pixels_per_axis"][0] * image_settings["N_pixels_per_axis"][1],
                      experiment_settings["latent_dimension"],
                      experiment_settings["encoder"]["hidden_dimensions"], network_type="encoder", device=device,
                      latent_type="categorical")
        decoder = MLP(1, experiment_settings["N_domains"]*6,
                      experiment_settings["decoder"]["hidden_dimensions"], network_type="decoder", device=device)

    vae = VAE(encoder, decoder, device, N_domains = experiment_settings["N_domains"], N_residues= experiment_settings["N_residues"],
              tau_mask=experiment_settings["tau_mask"], mask_start_values=experiment_settings["mask_start"],
              latent_type=experiment_settings["latent_type"], latent_dim=experiment_settings["latent_dimension"])
    vae.to(device)

    pixels_x = np.linspace(image_settings["image_lower_bounds"][0], image_settings["image_upper_bounds"][0],
                           num=image_settings["N_pixels_per_axis"][0]).reshape(1, -1)

    pixels_y = np.linspace(image_settings["image_lower_bounds"][1], image_settings["image_upper_bounds"][1],
                           num=image_settings["N_pixels_per_axis"][1]).reshape(1, -1)

    renderer = Renderer(pixels_x, pixels_y, N_atoms = experiment_settings["N_residues"]*3,
                        period=image_settings["renderer"]["period"], std=1, defocus=image_settings["renderer"]["defocus"],
                        spherical_aberration=image_settings["renderer"]["spherical_aberration"],
                        accelerating_voltage=image_settings["renderer"]["accelerating_voltage"],
                        amplitude_contrast_ratio=image_settings["renderer"]["amplitude_contrast_ratio"],
                        device=device, use_ctf=image_settings["renderer"]["use_ctf"],
                        latent_type=experiment_settings["latent_type"], latent_dim=experiment_settings["latent_dimension"])


    base_structure = read_pdb(experiment_settings["base_structure_path"])
    centering_structure = read_pdb(experiment_settings["centering_structure_path"])
    center_of_mass = compute_center_of_mass(centering_structure)
    centered_based_structure = center_protein(base_structure, center_of_mass)
    atom_positions = torch.tensor(get_backbone(centered_based_structure), dtype=torch.float32, device=device)

    if experiment_settings["optimizer"]["name"] == "adam":
        if "learning_rate_mask" not in experiment_settings["optimizer"]:
            optimizer = torch.optim.Adam(vae.parameters(), lr=experiment_settings["optimizer"]["learning_rate"])
        else:
            list_param = [{"params": param, "lr":experiment_settings["optimizer"]["learning_rate_mask"]} for name, param in
                          vae.named_parameters() if "mask" in name]
            list_param.append({"params": vae.encoder.parameters(), "lr":experiment_settings["optimizer"]["learning_rate"]})
            list_param.append({"params": vae.decoder.parameters(), "lr":experiment_settings["optimizer"]["learning_rate"]})
            optimizer = torch.optim.Adam(list_param)
    else:
        raise Exception("Optimizer must be Adam")

    dataset = ImageDataSet(experiment_settings["dataset_images_path"], experiment_settings["dataset_poses_path"],
                           experiment_settings["dataset_poses_translation_path"])

    N_epochs = experiment_settings["N_epochs"]
    batch_size = experiment_settings["batch_size"]
    latent_type = experiment_settings["latent_type"]
    assert latent_type in ["continuous", "categorical"]

    return vae, renderer, atom_positions, optimizer, dataset, N_epochs, batch_size, experiment_settings, latent_type, device


def monitor_training(mask, tracking_metrics, epoch, experiment_settings, vae):
    """
    Monitors the training process through wandb and saving masks and models
    :param mask:
    :param tracking_metrics:
    :param epoch:
    :param experiment_settings:
    :param vae:
    :return:
    """
    wandb.log({key: np.mean(val) for key, val in tracking_metrics.items()})
    wandb.log({"epoch": epoch})
    hard_mask = np.argmax(mask.detach().cpu().numpy(), axis=-1)
    for l in range(experiment_settings["N_domains"]):
        wandb.log({f"mask_{l}": np.sum(hard_mask[0] == l)})

    mask_python = mask.to("cpu").detach()
    np.save(experiment_settings["folder_path"] + "masks/mask" + str(epoch) + ".npy", mask_python[0])
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
    :return: structure object from BioPython
    """
    parser = PDBParser(PERMISSIVE=0)
    structure = parser.get_structure("A", path)
    return structure


def compute_rotations_per_residue(quaternions, mask, device):
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

def compute_translations_per_residue(translation_vectors, mask):
    """
    Computes one translation vector per residue based on the mask
    :param translation_vectors: torch.tensor (Batch_size, N_domains, 3) translations for each domain
    :param mask: torch.tensor(N_batch, N_residues, N_domains) weights of the attention mask
    :return: translation per residue torch.tensor(batch_size, N_residues, 3)
    """
    #### How is it possible to compute this product given the two tensor size
    translation_per_residue = torch.einsum("bij, bjk -> bik", mask, translation_vectors)
    return translation_per_residue


def deform_structure(atom_positions, translation_per_residue, rotations_per_residue):
    """
    Note that the reference frame absolutely needs to be the SAME for all the residues (placed in the same spot),
     otherwise the rotation will NOT be approximately rigid !!!
    :param positions: torch.tensor(N_residues*3, 3)
    :param translation_vectors: translations vectors:
            tensor (Batch_size, N_residues, 3)
    :param rotations_per_residue: tensor (N_batch, N_residues, 3, 3) of rotation matrices per residue
    :return: tensor (Batch_size, 3*N_residues, 3) corresponding to translated structure, tensor (3*N_residues, 3)
            of translation vectors
    """
    ## We displace the structure, using an interleave because there are 3 consecutive atoms belonging to one
    ## residue.
    ##We compute the rotated residues, where this axis of rotation is at the origin.
    rotation_per_atom = torch.repeat_interleave(rotations_per_residue, 3, dim=1)
    transformed_atom_positions = torch.einsum("lbjk, bk->lbj", rotation_per_atom, atom_positions)
    #transformed_atom_positions = torch.matmul(rotation_per_atom, atom_positions[None, :, :, None])
    new_atom_positions = transformed_atom_positions + torch.repeat_interleave(translation_per_residue,
                                                                                              3, 1)
    return new_atom_positions