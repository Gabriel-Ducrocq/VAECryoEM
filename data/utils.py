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

def compute_poses(structure_paths, poses, center_vector):
    """
    Rotate the pdb structures so that we have different poses
    :param dataset_path: list of str, path to the structures
    :param poses: np.array(N_structures, 3, 3) of rotation matrices
    :return: list of PDB structures
    """
    parser = PDBParser(PERMISSIVE=0)
    print("Reading pdb files:")
    list_structures = [parser.get_structure("A", file)
                       for i, file in tqdm.tqdm(enumerate(structure_paths))]
    print("Centering structures:")
    _ = [center_protein(struct, center_vector)
                       for i, struct in tqdm.tqdm(enumerate(list_structures))]

    print("Rotating structures")
    _ = [rotate_pdb_structure_matrix(struct, poses[i])
                       for i, struct in tqdm.tqdm(enumerate(list_structures))]

    return list_structures


def save_structures(list_structures, folder_path):
    """
    Save the structures into PDB files
    :param list_structures: list of BioPDB structures
    :param folder_path: str, path to the folder where we want to save the structures
    :return: None
    """
    io = PDBIO()
    for i, struct in enumerate(list_structures):
        io.set_structure(struct)
        io.save(folder_path + f"posed_structure_{i}.pdb")





