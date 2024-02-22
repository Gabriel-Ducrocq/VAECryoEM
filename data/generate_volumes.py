import sys
import os
path = os.path.abspath("model")
sys.path.append(path)
import yaml
import torch
import utils
import argparse
import mrcfile
import numpy as np
from tqdm import tqdm
from gmm import Gaussian, EMAN2Grid
from cryodrgn import mrc
from polymer import Polymer
from Bio.PDB import PDBParser
from renderer import structure_to_volume
from Bio import BiopythonWarning

parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--apix', type=float, required=True)
parser_arg.add_argument('--Npix', type=float, required=True)
parser_arg.add_argument('--folder_volumes', type=str, required=True)
parser_arg.add_argument('--folder_structures', type=str, required=True)
parser_arg.add_argument('--centering_structure', type=str, required=True)
parser_arg.add_argument('--batch_size', type=str, required=True)
args = parser_arg.parse_args()
apix = args.apix
folder_volumes = args.folder_volumes
folder_structures = args.folder_structures
batch_size = args.batch_size
centering_structure = args.centering_structure
Npix = args.Npix


grid = EMAN2Grid(Npix, apix)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

centering_structure = Polymer.from_pdb(centering_structure)
centering_vector = np.mean(centering_structure.coord, axis=0)
structures_poly = [Polymer.from_pdb(folder_structures + path) for path in os.listdir(folder_structures) if ".pdb" in path]
structures = [Gaussian(torch.tensor(poly.coord - centering_vector - apix/2), torch.tensor([[2]*centering_vector.shape[0]]), poly.num_electron)
                for poly in structures_poly]

print("center", np.mean(structures_poly[0].coord - centering_vector, axis=0))
print("min coord, max_coord", torch.min(grid.line_coords), torch.max(grid.line_coords))
name = "test"
N = len(structures)
for i in tqdm(range(0,N)):
    batch_struct = structures[i]
    batch_volumes = structure_to_volume(torch.tensor(batch_struct.mus)[None, :, :], torch.tensor(batch_struct.sigmas), torch.tensor(batch_struct.amplitudes)[:, None], grid.line_coords)

    #mrc.write(f"{folder_volumes}volume_{indexes[i]}.mrc", np.transpose(batch_volumes[0].detach().cpu().numpy(), axes=(2, 1, 0)), Apix=1.0, is_vol=True)
    print(batch_volumes.shape)
    mrc.write(f"{folder_volumes}volume_{name}13.mrc", np.transpose(batch_volumes[0].detach().cpu().numpy(), axes=(2, 1, 0)), Apix=1.0, is_vol=True)
    #mrc.write(f"{folder_volumes}volume_{name}4.mrc", batch_volumes[0].detach().cpu().numpy(), Apix=1.0, is_vol=True)

    print("\n\n\n")
    #with mrcfile.new(f"{folder_volumes}volume_{i}.mrc", overwrite=True) as mrc:
    #	print(batch_volumes.shape)
    # 	print(np.sum(batch_volumes[0,:, :, 0].detach().numpy()))
    #	mrc.set_data(batch_volumes[0,:, :, :].detach().numpy())
    #	mrc.header["origin"]["x"] = -95
    #	mrc.header["origin"]["y"] = -95
    #	mrc.header["origin"]["z"] = -95

    #	mrc.header["nxstart"] = -95
    #	mrc.header["nystart"] = -95
    #	mrc.header["nzstart"] = -95
    #	print(mrc.header["origin"])




