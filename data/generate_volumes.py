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
parser_arg.add_argument('--thinning', type=int, required=True)
parser_arg.add_argument('--folder_volumes', type=str, required=True)
parser_arg.add_argument('--folder_structures', type=str, required=True)
parser_arg.add_argument('--batch_size', type=str, required=True)
parser_arg.add_argument('--base_structure', type=str, required=True)
parser_arg.add_argument('--center', action=argparse.BooleanOptionalAction)
args = parser_arg.parse_args()
apix = args.apix
folder_volumes = args.folder_volumes
folder_structures = args.folder_structures
batch_size = args.batch_size
base_structure = args.base_structure
center = args.center
Npix = args.Npix
thinning = args.thinning


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
grid = EMAN2Grid(Npix, apix, device=device)
filter_aa = True
base_structure = Polymer.from_pdb(base_structure, filter_aa)
indexes = [path.split("_")[-1].split(".")[0] for path in os.listdir(folder_structures) if ".pdb" in path]
print("Indexes", indexes)
print("Number indexes", len(indexes))
paths = [folder_structures + path for path in os.listdir(folder_structures) if ".pdb" in path]
sorted_paths = list(zip(*sorted(zip(indexes, paths))))[-1]
print("Structures list", sorted_paths)
sorted_paths = sorted_paths[::thinning]
print("Reading pdbs")
structures_poly = [Polymer.from_pdb(path) for path in sorted_paths]
if not center:
    print("No centering")
    structures = [Gaussian(torch.tensor(poly.coord), torch.tensor([[2]*poly.coord.shape[0]]), base_structure.num_electron)
                    for poly in structures_poly]
else:
    print("Centering")
    center_vector = np.mean(base_structure.coord, axis=0)[None, :]
    structures = [Gaussian(torch.tensor(poly.coord - center_vector), torch.tensor([[2]*poly.coord.shape[0]]), base_structure.num_electron)
                    for poly in structures_poly]


print("min coord, max_coord", torch.min(grid.line_coords), torch.max(grid.line_coords))
N = len(structures)
for i in tqdm(range(0,N)):
    batch_struct = structures[i]
    batch_volumes = structure_to_volume(torch.tensor(batch_struct.mus).to(device)[None, :, :], torch.tensor(batch_struct.sigmas).to(device), torch.tensor(batch_struct.amplitudes).to(device)[:, None], grid, device)

    #mrc.write(f"{folder_volumes}volume_{indexes[i]}.mrc", np.transpose(batch_volumes[0].detach().cpu().numpy(), axes=(2, 1, 0)), Apix=1.0, is_vol=True)
    print(batch_volumes.shape)
    mrc.write(f"{folder_volumes}volume_{i}.mrc", np.transpose(batch_volumes[0].detach().cpu().numpy(), axes=(2, 1, 0)), Apix=1.0, is_vol=True)
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




