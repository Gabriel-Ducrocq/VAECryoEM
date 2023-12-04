mport matplotlib.pyplot as plt
import yaml
import torch
import numpy as np
import Bio.PDB as bpdb
from Bio.PDB import PDBIO
from Bio.PDB.PDBParser import PDBParser
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from protein.main import rotate_pdb_structure_matrix, rotate_residues


path = os.path.abspath("model")
sys.path.append(path)