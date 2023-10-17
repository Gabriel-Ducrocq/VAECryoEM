import torch
import yaml
import numpy as np
from Bio.PDB.PDBParser import PDBParser
from pytorch3d.transforms import quaternion_to_axis_angle, axis_angle_to_matrix



def parse_yaml(path):
    """
    Parse the yaml file to get the setting for the run
    :param path: str, path to the yaml file
    :return: dict, settings for the run
    """
    with open(path, "r") as file:
        experiment_settings = yaml.safe_load(file)

    return experiment_settings



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
                if name != "LBV":
                    x, y, z = get_atom_positions(residue, name)
                    absolute_positions.append(x)
                    absolute_positions.append(y)
                    absolute_positions.append(z)

                    N_residue += 1

    return np.array(absolute_positions)


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
    :param mask: tensor (N_residues, N_domains)
    :return: tensor (N_batch, N_residues, 3, 3) rotation matrix for each residue
    """
    N_residues = mask.shape[0]
    batch_size = quaternions.shape[0]
    N_domains = mask.shape[-1]
    # NOTE: no need to normalize the quaternions, quaternion_to_axis does it already.
    rotation_per_domains_axis_angle = quaternion_to_axis_angle(quaternions)
    mask_rotation_per_domains_axis_angle = mask[None, :, :, None] * rotation_per_domains_axis_angle[:, None, :, :]

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
    :param mask: torch.tensor(N_residues, N_domains) weights of the attention mask
    :return: translation per residue torch.tensor(batch_size, N_residues, 3)
    """
    #### How is it possible to compute this product given the two tensor size
    translation_per_residue = torch.matmul(mask, translation_vectors)
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
    transformed_atom_positions = torch.matmul(rotation_per_atom, atom_positions[None, :, :, None])
    new_atom_positions = transformed_atom_positions[:, :, :, 0] + torch.repeat_interleave(translation_per_residue,
                                                                                              3, 1)
    return new_atom_positions