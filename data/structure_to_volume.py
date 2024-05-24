import sys
import os
path = os.path.abspath("model")
sys.path.append(path)
import mrc
import yaml
import torch
import utils
import mrcfile
import argparse
import numpy as np
import renderer
import time
from polymer import Polymer
from gmm import Gaussian, EMAN2Grid


filter_aa = True


def structure_to_volume(image_yaml, structure_path, output_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(image_yaml, "r") as file:
        image_settings = yaml.safe_load(file)

    apix = image_settings["apix"]
    Npix = image_settings["Npix"]
    Npix_downsize = image_settings["Npix_downsize"]
    apix_downsize = apix*Npix/Npix_downsize

    filter_aa = True
    base_structure = Polymer.from_pdb(structure_path, filter_aa)
    amplitudes = torch.tensor(base_structure.num_electron, dtype=torch.float32, device=device)[:, None]
    grid = EMAN2Grid(Npix_downsize, apix_downsize, device=device)
    gmm_repr = Gaussian(torch.tensor(base_structure.coord, dtype=torch.float32, device=device), 
            torch.ones((base_structure.coord.shape[0], 1), dtype=torch.float32, device=device)*image_settings["sigma_gmm"], 
            amplitudes)
    start = time.time()
    struct = gmm_repr.mus[None, :, :].repeat(128, 1, 1)
    volume = renderer.structure_to_volume(struct, gmm_repr.sigmas, gmm_repr.amplitudes, grid, device)
    end = time.time()
    print(f"Time to generate a volume on {Npix_downsize} a side:", end-start)
    start = time.time()
    volume = renderer.structure_to_volume(struct, gmm_repr.sigmas, gmm_repr.amplitudes, grid, device)
    end = time.time()
    print(f"Time to generate a volume on {Npix_downsize} a side again :", end-start)
    start = time.time()
    volume = renderer.structure_to_volume(struct, gmm_repr.sigmas, gmm_repr.amplitudes, grid, device)
    end = time.time()
    print(f"Time to generate a volume on {Npix_downsize} a side again2:", end-start)
    start = time.time()
    volume = renderer.structure_to_volume(struct, gmm_repr.sigmas, gmm_repr.amplitudes, grid, device)
    end = time.time()
    print(f"Time to generate a volume on {Npix_downsize} a side again3:", end-start)
    mrc.MRCFile.write(output_path, np.transpose(volume[0].detach().cpu().numpy(), axes=(2, 1, 0)), Apix=apix_downsize, is_vol=True)


if __name__ == '__main__':
    parser_arg = argparse.ArgumentParser()
    parser_arg.add_argument('--image_yaml', type=str, required=True)
    parser_arg.add_argument("--structure_path", type=str, required=True)
    parser_arg.add_argument("--output_path", type=str, required=True)
    args = parser_arg.parse_args()
    image_yaml = args.image_yaml
    structure_path = args.structure_path
    output_path = args.output_path
    structure_to_volume(image_yaml, structure_path, output_path)




