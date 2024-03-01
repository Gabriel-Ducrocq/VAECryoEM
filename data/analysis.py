import sys
import os
path = os.path.abspath("model")
sys.path.append(path)
import yaml
import torch
import utils
import mrcfile
import renderer
import argparse
import starfile
import numpy as np
from ctf import CTF
from tqdm import tqdm
import Bio.PDB as bpdb
from Bio.PDB import PDBIO
from polymer import Polymer
from Bio.PDB import PDBParser
from dataset import ImageDataSet
from gmm import Gaussian, EMAN2Grid
from torch.utils.data import DataLoader
from pytorch3d.transforms import quaternion_to_axis_angle
from protein.main import rotate_residues, translate_residues

class ResSelect(bpdb.Select):
    def accept_residue(self, res):
        if res.get_resname() == "LBV":
            return False
        else:
            return True

def concat_and_save(tens, path):
    """
    Concatenate the lsit of tensor along the dimension 0
    :param tens: list of tensor with batch size as dim 0
    :param path: str, path to save the torch tensor
    :return: tensor of concatenated tensors
    """
    concatenated = torch.concat(tens, dim=0)
    np.save(path, concatenated.detach().numpy())
    return concatenated


parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--folder_experiment', type=str, required=True)
parser_arg.add_argument("--model", type=str, required=True)
parser_arg.add_argument("--folder_output", type=str, required=True)
parser_arg.add_argument("--type", type=str, required=True)
parser_arg.add_argument("--step", type=int, required=True)
args = parser_arg.parse_args()
folder_experiment = args.folder_experiment
folder_output = args.folder_output
output_type = args.type
model_path = args.model
step = args.step
batch_size = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(f"{folder_experiment}/parameters.yaml", "r") as file:
    experiment_settings = yaml.safe_load(file)

with open(f"{folder_experiment}/images.yaml", "r") as file:
    image_settings = yaml.safe_load(file)


apix = image_settings["apix"]
Npix = image_settings["Npix"]
base_structure = Polymer.from_pdb(experiment_settings["base_structure_path"])
centering_structure = Polymer.from_pdb(experiment_settings["centering_structure_path"])
center_of_mass = utils.compute_center_of_mass(centering_structure)
# Since we operate on an EMAN2 grid, we need to translate the structure by -apix/2 to get it at the center of the image.
base_structure.translate_structure(-center_of_mass - apix/2)
gmm_repr = Gaussian(torch.tensor(base_structure.coord, dtype=torch.float32, device=device), 
            torch.ones((base_structure.coord.shape[0], 1), dtype=torch.float32, device=device)*image_settings["sigma_gmm"], 
            torch.tensor(base_structure.num_electron, dtype=torch.float32, device=device)[:, None]) 

particles_star = starfile.read(experiment_settings["star_file"])
particles_mrcs = experiment_settings["mrcs_file"]
with mrcfile.open(particles_mrcs) as f:
    images = f.data


images = torch.tensor(np.stack(images, axis = 0), dtype=torch.float32)
ctf_experiment = CTF.from_starfile(experiment_settings["star_file"], device=device)

dataset = ImageDataSet(images, particles_star["particles"]) 

if device == "cpu":
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.device = "cpu"
else:
    model = torch.load(model_path, map_location=torch.device(device))
    model.device = device

model.eval()

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

parser = PDBParser(PERMISSIVE=0)

identity_pose = torch.broadcast_to(torch.eye(3,3, device=device)[None, :, :], (batch_size, 3, 3))
zeros_poses_translation = torch.broadcast_to(torch.zeros((3,), device=device)[None, :], (batch_size, 3))

all_latent_mean = []
all_latent_std = []
all_rotations_per_residue = []
all_translation_per_residue = []
all_translation_per_domain = []
all_axis_angle_per_domain = []



### BE CAREFUL !!!! ADDED A START !!!!
#start = step*5000
start = 0


images = data_loader.images[start::step]
#for i, (batch_images, batch_poses, batch_poses_translation) in tqdm(enumerate(data_loader)):
for i, batch_images in tqdm(enumerate(images)):
    batch_images = batch_images[None, :]
    batch_poses = batch_poses[None, :, :]
    batch_poses_translation = batch_poses_translation[None, :]
    #print("Batch number:", i)
    batch_images = batch_images.to(device)
    latent_variables, latent_mean, latent_std = model.sample_latent(batch_images)
    """
    mask = vae.sample_mask(batch_size)
    quaternions_per_domain, translations_per_domain = vae.decode(latent_variables)
    rotation_per_residue = model.utils.compute_rotations_per_residue(quaternions_per_domain, mask, device)
    translation_per_residue = model.utils.compute_translations_per_residue(translations_per_domain, mask)
    predicted_structures = model.utils.deform_structure(gmm_repr.mus, translation_per_residue,
                                                       rotation_per_residue)

    if output_type == "images":
            predicted_images = renderer.project(posed_predicted_structures, gmm_repr.sigmas, gmm_repr.amplitudes, grid)
            batch_predicted_images = renderer.apply_ctf(predicted_images, ctf, indexes)
            batch_predicted_images = torch.flatten(batch_predicted_images, start_dim=-2, end_dim=-1)
            batch_predicted_images = dataset.standardize(batch_predicted_images, device=device)

        np.save(f"{folder_output}predicted_images_{i+ start}.npy", batch_predicted_images.to("cpu").detach().numpy())

    ### THE VOLUMES CREATION IS WRONG !!!!! ########
    if output_type == "volumes":
        batch_predicted_volumes = renderer_no_ctf.compute_x_y_values_all_atoms(deformed_structures, identity_pose, zeros_poses_translation, 
            latent_type=experiment_settings["latent_type"], volume=True)

        mrc.write(f"{folder_output}volume_{i+start}.mrc", np.transpose(batch_predicted_volumes[0].detach().cpu().numpy(), axes=(2, 1, 0)), Apix=1.0, is_vol=True)
    """
    all_latent_mean.append(latent_mean.to("cpu"))
    all_latent_std.append(latent_std.to("cpu"))
    np.save(f"{folder_output}all_rotations_per_residue_{i}.npy", rotation_per_residue.to("cpu").detach().numpy())
    np.save(f"{folder_output}all_translation_per_residue_{i}.npy", translation_per_residue.to("cpu").detach().numpy())
    #all_translation_per_residue.append(translation_per_residue.to("cpu"))
    #all_rotations_per_residue.append(rotation_per_residue.to("cpu"))
    #all_axis_angle_per_domain.append(axis_angle_per_domain.to("cpu"))
    #all_translation_per_domain.append(translations_per_domain.to("cpu"))




#all_rotations_per_residue = concat_and_save(all_rotations_per_residue, f"{folder_output}all_rotations_per_residue.npy")
#all_translation_per_residue = concat_and_save(all_translation_per_residue, f"{folder_output}all_translation_per_residue.npy")
all_latent_mean = concat_and_save(all_latent_mean, f"{folder_output}all_latent_mean.npy")
all_latent_std = concat_and_save(all_latent_std, f"{folder_output}all_latent_std.npy")

#all_rotations_per_domain = concat_and_save(all_axis_angle_per_domain, f"{folder_experiment}all_rotations_per_domain.npy")
#all_translation_per_domain = concat_and_save(all_translation_per_domain, f"{folder_experiment}all_translation_per_domain.npy")
#print("REGISTERED !")


#all_rotations_per_residue = np.load(f"{folder_output}all_rotations_per_residue.npy")
#all_translation_per_residue = np.load(f"{folder_output}all_translation_per_residue.npy")

all_rotations_per_residue = []
all_translation_per_residue = []
for i in range(10000):
    all_rotations_per_residue.append(np.load(f"{folder_output}all_rotations_per_residue_{i}.npy"))
    all_translation_per_residue.append(np.load(f"{folder_output}all_translation_per_residue_{i}.npy"))

all_rotations_per_residue = np.concatenate(all_rotations_per_residue, axis=0)
all_translation_per_residue = np.concatenate(all_translation_per_residue, axis=0)


#for i in range(all_translation_per_residue.shape[0]):
for i in tqdm(range(6397, 10000)):
    print("Deform structure:", i)
    a = torch.ones((4,), device=device)
    a += torch.ones((4,), device=device)
    parser = PDBParser(PERMISSIVE=0)
    structure = utils.read_pdb(experiment_settings["base_structure_path"])
    io = PDBIO()
    io.set_structure(structure)
    io.save(f"{folder_output}predicted_structures/predicted_structure_{i+1}.pdb", ResSelect())
    structure = utils.read_pdb(f"{folder_output}predicted_structures/predicted_structure_{i+1}.pdb")
    structure = utils.center_protein(structure, center_of_mass[0])
    rotate_residues(structure, all_rotations_per_residue[i], np.eye(3,3))
    translate_residues(structure, all_translation_per_residue[i])
    structure = utils.center_protein(structure, -center_of_mass[0])
    io = PDBIO()
    io.set_structure(structure)
    io.save(f"{folder_output}predicted_structures/predicted_structure_{i+1}.pdb")





N_images = experiment_settings["N_images"]
#for i in range(N_images):











