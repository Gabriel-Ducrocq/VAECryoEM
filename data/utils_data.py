import tqdm
import numpy as np
from Bio.PDB import PDBIO
from Bio.PDB.PDBParser import PDBParser


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

def rotate_pdb_structure_matrix(pdb_structure, rotation_matrix):
    """
    Rotates the entire structure according to a rotation matrix.
    :param pdb_structure: PDB structure to rotate
    :param rotation_matrix: numpy.array (3, 3) matrix of rotation
    """
    all_coords = np.concatenate([atom.coord[0, :, None] for atom in pdb_structure.get_atoms()], axis=1)
    rotated_coordinates = np.matmul(rotation_matrix, all_coords)
    for index, atom in enumerate(pdb_structure.get_atoms()):
        atom.set_coord(rotated_coordinates[:, index])

def translate_pdb_structure(pdb_structure, translation_vector):
    """
    Translate the entire structure according to a translation vector
    :param pdb_structure: PDB structure to translate
    :param translation_vector: numpy.array (3, ) vector of translation
    """
    all_coords = np.concatenate([atom.coord[0, :, None] for atom in pdb_structure.get_atoms()], axis=1)
    translated_coordinates = all_coords - translation_vector
    for index, atom in enumerate(pdb_structure.get_atoms()):
        atom.set_coord(translated_coordinates[:, index])

def compute_poses(structure_path, pose, translation, center_vector):
    """
    Rotate the pdb structure so that we have it with a pose
    :param dataset_path: str, path to the structure
    :param pose: np.array(3, 3) of rotation matrix
    :param translation: np.array(3,) of translation vector
    :return: PDB structure
    """
    parser = PDBParser(PERMISSIVE=0)
    struct = parser.get_structure("A", structure_path)
    center_protein(struct, center_vector)
    rotate_pdb_structure_matrix(struct, pose)
    return struct


def save_structure(structure, path):
    """
    Save the structures into PDB files
    :param structure: BioPDB structures
    :param folder_path: str, path to the folder where we want to save the structure
    :return: None
    """
    io = PDBIO()
    io.set_structure(structure)
    io.save(path)
