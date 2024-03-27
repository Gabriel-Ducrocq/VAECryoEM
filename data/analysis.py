import sys
import os
path = os.path.abspath("model")
sys.path.append(path)
import torch
import yaml
import utils
import argparse
import numpy as np
from tqdm import tqdm
import Bio.PDB as bpdb
from cryodrgn import mrc
from Bio.PDB import PDBIO
from protein.main import rotate_residues, translate_residues
from Bio.PDB import PDBParser
from renderer import Renderer
from dataset import ImageDataSet
from torch.utils.data import DataLoader
from pytorch3d.transforms import quaternion_to_axis_angle

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

pixels_x = np.linspace(image_settings["image_lower_bounds"][0], image_settings["image_upper_bounds"][0],
                       num=image_settings["N_pixels_per_axis"][0]).reshape(1, -1)

pixels_y = np.linspace(image_settings["image_lower_bounds"][1], image_settings["image_upper_bounds"][1],
                       num=image_settings["N_pixels_per_axis"][1]).reshape(1, -1)

renderer_no_ctf = Renderer(pixels_x, pixels_y, N_atoms=experiment_settings["N_residues"] * 3,
                    dfU=image_settings["renderer"]["dfU"], dfV=image_settings["renderer"]["dfV"],
                    dfang=image_settings["renderer"]["dfang"],
                    spherical_aberration=image_settings["renderer"]["spherical_aberration"],
                    accelerating_voltage=image_settings["renderer"]["accelerating_voltage"],
                    amplitude_contrast_ratio=image_settings["renderer"]["amplitude_contrast_ratio"],
                    device=device, use_ctf=image_settings["renderer"]["use_ctf"])

if device == "cpu":
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.device = "cpu"
else:
    model = torch.load(model_path, map_location=torch.device(device))
    model.device = device

model.eval()


images_path = torch.load(f"{folder_experiment}ImageDataSet")
poses = torch.load(f"{folder_experiment}poses")
poses_translations = torch.load(f"{folder_experiment}poses_translation")
dataset = ImageDataSet(experiment_settings["dataset_images_path"], experiment_settings["dataset_poses_path"],
                       experiment_settings["dataset_poses_translation_path"])
data_loader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False))

parser = PDBParser(PERMISSIVE=0)
base_structure = utils.read_pdb(experiment_settings["base_structure_path"])
centering_structure = utils.read_pdb(experiment_settings["centering_structure_path"])
center_of_mass = utils.compute_center_of_mass(centering_structure)
centered_based_structure = utils.center_protein(base_structure, center_of_mass)
atom_positions = torch.tensor(utils.get_backbone(centered_based_structure), dtype=torch.float32, device=device)
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


images = torch.flatten(torch.load(experiment_settings["dataset_images_path"])[start::step],start_dim=-2, end_dim=-1)
iterable = zip(images, torch.load(experiment_settings["dataset_poses_path"])[start::step], 
torch.load(experiment_settings["dataset_poses_translation_path"])[start::step])
#for i, (batch_images, batch_poses, batch_poses_translation) in tqdm(enumerate(data_loader)):
for i, (batch_images, batch_poses, batch_poses_translation) in tqdm(enumerate(iterable)):
    batch_images = batch_images[None, :]
    batch_poses = batch_poses[None, :, :]
    batch_poses_translation = batch_poses_translation[None, :]
    #print("Batch number:", i)
    batch_images = batch_images.to(device)
    batch_poses = batch_poses.to(device)
    batch_poses_translation = batch_poses_translation.to(device)
    latent_variables, latent_mean, latent_std = model.sample_latent(batch_images)
    mask = model.sample_mask(N_batch=batch_size)
    quaternions_per_domain, translations_per_domain = model.decode(latent_mean)
    axis_angle_per_domain = quaternion_to_axis_angle(quaternions_per_domain)
    rotation_per_residue = utils.compute_rotations_per_residue(quaternions_per_domain, mask, device)
    translation_per_residue = utils.compute_translations_per_residue(translations_per_domain, mask)
    deformed_structures = utils.deform_structure(atom_positions, translation_per_residue,
                                                       rotation_per_residue)

    if output_type == "images":
        batch_predicted_images = renderer_no_ctf.compute_x_y_values_all_atoms(deformed_structures, identity_pose,
                                            zeros_poses_translation, latent_type=experiment_settings["latent_type"])
        np.save(f"{folder_output}predicted_images_{i+ start}.npy", batch_predicted_images.to("cpu").detach().numpy())

    if output_type == "volumes":
        batch_predicted_volumes = renderer_no_ctf.compute_x_y_values_all_atoms(deformed_structures, identity_pose, zeros_poses_translation, 
            latent_type=experiment_settings["latent_type"], volume=True)

        mrc.write(f"{folder_output}volume_{i+start}.mrc", np.transpose(batch_predicted_volumes[0].detach().cpu().numpy(), axes=(2, 1, 0)), Apix=1.0, is_vol=True)

    #all_latent_mean.append(latent_mean.to("cpu"))
    #all_latent_std.append(latent_std.to("cpu"))
    np.save(f"{folder_output}all_rotations_per_residue_{i}.npy", rotation_per_residue.to("cpu").detach().numpy())
    np.save(f"{folder_output}all_translation_per_residue_{i}.npy", translation_per_residue.to("cpu").detach().numpy())
    #all_translation_per_residue.append(translation_per_residue.to("cpu"))
    #all_rotations_per_residue.append(rotation_per_residue.to("cpu"))
    #all_axis_angle_per_domain.append(axis_angle_per_domain.to("cpu"))
    #all_translation_per_domain.append(translations_per_domain.to("cpu"))




#all_rotations_per_residue = concat_and_save(all_rotations_per_residue, f"{folder_output}all_rotations_per_residue.npy")
#all_translation_per_residue = concat_and_save(all_translation_per_residue, f"{folder_output}all_translation_per_residue.npy")
#all_latent_mean = concat_and_save(all_latent_mean, f"{folder_output}all_latent_mean.npy")
#all_latent_std = concat_and_save(all_latent_std, f"{folder_output}all_latent_std.npy")

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
for i in tqdm(range(0, 10000)):
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











