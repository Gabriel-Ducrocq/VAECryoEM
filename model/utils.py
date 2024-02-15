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
from pytorch3d.transforms import quaternion_to_axis_angle, axis_angle_to_matrix, rotation_6d_to_matrix, matrix_to_axis_angle


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

    if experiment_settings["mask_prior"]["type"] == "uniform":
        experiment_settings["mask_prior"] = compute_mask_prior(experiment_settings["N_residues"],
                                                               experiment_settings["N_domains"], device)
    else:
        for mask_prior_key in experiment_settings["mask_prior"].keys():
            experiment_settings["mask_prior"][mask_prior_key]["mean"] = torch.tensor(experiment_settings["mask_prior"][mask_prior_key]["mean"],
                                                                                     dtype=torch.float32, device=device)
            experiment_settings["mask_prior"][mask_prior_key]["std"] = torch.tensor(experiment_settings["mask_prior"][mask_prior_key]["std"],
                                                                                     dtype=torch.float32, device=device)

    if experiment_settings["resume_training"]["model"] == "None":
        vae = VAE(device, N_domains = experiment_settings["N_domains"], N_residues= experiment_settings["N_residues"],
                  tau_mask=experiment_settings["tau_mask"], mask_start_values=experiment_settings["mask_start"],
                  latent_type=experiment_settings["latent_type"], latent_dim=experiment_settings["latent_dimension"],
                   N_images =experiment_settings["N_images"], representation=experiment_settings["representation"])
        vae.to(device)
    else:
        vae = torch.load(experiment_settings["resume_training"]["model"])
        vae.to(device)

    #Note that in this configuration the pixels are not really centered on "round" value, e.g
    #if we take linsapce(-10, 10, 20) the pixels won't be centered at -9, -8 and so on.
    pixels_x = np.linspace(image_settings["image_lower_bounds"][0], image_settings["image_upper_bounds"][0],
                           num=image_settings["N_pixels_per_axis"][0]).reshape(1, -1)

    pixels_y = np.linspace(image_settings["image_lower_bounds"][1], image_settings["image_upper_bounds"][1],
                           num=image_settings["N_pixels_per_axis"][1]).reshape(1, -1)

    renderer = Renderer(pixels_x, pixels_y, N_atoms=experiment_settings["N_residues"] * 3,
                        dfU=image_settings["renderer"]["dfU"], dfV=image_settings["renderer"]["dfV"],
                        dfang=image_settings["renderer"]["dfang"],
                        spherical_aberration=image_settings["renderer"]["spherical_aberration"],
                        accelerating_voltage=image_settings["renderer"]["accelerating_voltage"],
                        amplitude_contrast_ratio=image_settings["renderer"]["amplitude_contrast_ratio"],
                        device=device, use_ctf=image_settings["renderer"]["use_ctf"], std = image_settings["renderer"]["std_volume"] if "std_volume" in image_settings["renderer"] else 1)

    base_structure = read_pdb(experiment_settings["base_structure_path"])
    centering_structure = read_pdb(experiment_settings["centering_structure_path"])
    center_of_mass = compute_center_of_mass(centering_structure)
    centered_based_structure = center_protein(base_structure, center_of_mass)
    atom_positions = torch.tensor(get_backbone(centered_based_structure), dtype=torch.float32, device=device)

    if experiment_settings["optimizer"]["name"] == "adam":
        if "learning_rate_mask" not in experiment_settings["optimizer"]:
            optimizer = torch.optim.Adam(vae.parameters(), lr=experiment_settings["optimizer"]["learning_rate"])
            pass
        else:
            list_param = []
            list_param.append({"params":vae.translation_per_domain, "lr":experiment_settings["optimizer"]["learning_rate"]})
            list_param.append({"params":vae.rotation_per_domain, "lr":experiment_settings["optimizer"]["learning_rate"]})

            list_param += [{"params": param, "lr":experiment_settings["optimizer"]["learning_rate_mask"]} for name, param in
                          vae.named_parameters() if "mask" in name]

            #optimizer = torch.optim.Adam(list_param)
            optimizer = torch.optim.Adam(list_param)
    else:
        raise Exception("Optimizer must be Adam")

    dataset = ImageDataSet(experiment_settings["dataset_images_path"], experiment_settings["dataset_poses_path"],
                           experiment_settings["dataset_poses_translation_path"])

    scheduler = None
    if "scheduler" in experiment_settings:
        milestones = experiment_settings["scheduler"]["milestones"]
        decay = experiment_settings["scheduler"]["decay"]
        print(f"Using MultiStepLR scheduler with milestones: {milestones} and decay factor {decay}.")
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=decay)

    N_epochs = experiment_settings["N_epochs"]
    batch_size = experiment_settings["batch_size"]
    latent_type = experiment_settings["latent_type"]
    N_domains = experiment_settings["N_domains"]
    N_images = experiment_settings["N_images"]
    assert latent_type in ["continuous", "categorical"]

    return vae, renderer, atom_positions, optimizer, dataset, N_epochs, batch_size, experiment_settings, latent_type, device, scheduler, N_images, N_domains


def compute_mask_prior(N_residues, N_domains, device):
    """
    Computes the mask prior if "uniform" is set in the yaml file
    :param N_residues: integer, number of residues
    :param N_domains: integer, number of domains
    :param device: str, device to use
    :return:
    """
    bound_0 = N_residues / N_domains
    mask_means_mean = torch.tensor(np.array([bound_0 / 2 + i * bound_0 for i in range(N_domains)]), dtype=torch.float32,
                          device=device)[None, :]

    mask_means_std = torch.tensor(np.ones(N_domains) * 10.0, dtype=torch.float32, device=device)[None, :]

    mask_stds_mean = torch.tensor(np.ones(N_domains) * bound_0, dtype=torch.float32, device=device)[None, :]

    mask_stds_std = torch.tensor(np.ones(N_domains) * 10.0, dtype=torch.float32, device=device)[None, :]

    mask_proportions_mean = torch.tensor(np.ones(N_domains) * 0, dtype=torch.float32, device=device)[None, :]

    mask_proportions_std = torch.tensor(np.ones(N_domains), dtype=torch.float32, device=device)[None, :]

    mask_prior = {}
    mask_prior["means"] = {"mean":mask_means_mean, "std":mask_means_std}
    mask_prior["stds"] = {"mean":mask_stds_mean, "std":mask_stds_std}
    mask_prior["proportions"] = {"mean":mask_proportions_mean, "std":mask_proportions_std}

    return mask_prior

def monitor_training(mask, tracking_metrics, epoch, experiment_settings, vae, optimizer):
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
    wandb.log({"lr":optimizer.param_groups[0]['lr']})
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


def compute_rotations_per_residue(r6, mask, device, representation = "r6"):
    """
    Computes the rotation matrix corresponding to each domain for each residue, where the angle of rotation has been
    weighted by the mask value of the corresponding domain.
    :param r6: tensor (N_batch, N_domains, 6) of r6 rotation representation
    :param mask: tensor (N_batch, N_residues, N_domains)
    :return: tensor (N_batch, N_residues, 3, 3) rotation matrix for each residue
    """
    N_residues = mask.shape[1]
    batch_size = r6.shape[0]
    N_domains = mask.shape[-1]
    # NOTE: no need to normalize the quaternions, quaternion_to_axis does it already.
    assert representation in ["r6", "axis_angle"], print("Rotation must represented as r6 of axis angle")
    if representation == "r6":
        rotation_per_domains_axis_angle = matrix_to_axis_angle(rotation_6d_to_matrix(r6))
    else:
        rotation_per_domains_axis_angle = r6

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


def add_noise(vae, lr_transformation, lr_mask, N_imagesN_domains, device):
    """
    Add noise to the gradient of the parameters
    """
    if vae.representation =="r6":
        d = 6
    else:
        d = 3
    noise_rot = torch.zeros(size=(N_images, N_domains, d), dtype=torch.float32, device=device)
    noise_trans = torch.zeros(size=(N_images, N_domains, 3), dtype=torch.float32, device=device)
    noise_rot[indexes_py] = torch.randn(size=(batch_size, N_domains, d), dtype=torch.float32, device=device)*np.sqrt(2/lr)
    noise_trans[indexes_py] = torch.randn(size=(batch_size, N_domains, 3), dtype=torch.float32, device=device)*np.sqrt(2/lr)
    loss.backward()
    vae.translation_per_domain.grad += noise_trans
    vae.rotation_per_domain.grad += noise_rot




