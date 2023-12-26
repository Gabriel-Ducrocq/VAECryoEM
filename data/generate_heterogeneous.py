import torch
import pickle
import argparse
import numpy as np

parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--folder_experiment', type=str, required=True)
parser_arg.add_argument('--folder_structures', type=str, required=True)
parser_arg.add_argument('--Apix', type=float, required=True)
args = parser_arg.parse_args()
folder_experiment = args.folder_experiment
folder_structures = args.folder_structures
N_pose_per_struct = args.N_pose_per_struct

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(f"{folder_experiment}/parameters.yaml", "r") as file:
    experiment_settings = yaml.safe_load(file)

with open(f"{folder_experiment}/images.yaml", "r") as file:
    image_settings = yaml.safe_load(file)

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
                    device=device, use_ctf=image_settings["renderer"]["use_ctf"])


N_pix = image_settings["N_pixels_per_axis"][0]
noise_var = image_settings["noise_var"]
centering_structure_path = experiment_settings["centering_structure_path"]
parser = PDBParser(PERMISSIVE=0)
centering_structure = parser.get_structure("A", centering_structure_path)

#Get all the structure and sort their names to have them in the right order.
structures = [folder_structures + path for path in os.listdir(folder_structures) if ".pdb" in path]
indexes = [int(name.split("/")[-1].split(".")[0].split("_")[-1]) for name in structures]
sorted_structures = [struct for _, struct in sorted(zip(indexes, structures))]
sorted_structures = [struct for struct in sorted_structures for _ in range(N_pose_per_struct)]


#Create poses:
N_images = experiments_settings["N_images"]*experiment_settings["N_pose_per_structure"]
axis_rotation = torch.randn((N_images, 3))
norm_axis = torch.sqrt(torch.sum(axis_rotation**2, dim=-1))
normalized_axis = axis_rotation/norm_axis[:, None]
print("Min norm of rotation axis", torch.min(torch.sqrt(torch.sum(normalized_axis**2, dim=-1))))
print("Max norm of rotation axis", torch.max(torch.sqrt(torch.sum(normalized_axis**2, dim=-1))))

angle_rotation = torch.rand((N_images,1))*torch.pi
plt.hist(angle_rotation[:, 0].detach().numpy())
plt.show()

axis_angle = normalized_axis*angle_rotation
poses = axis_angle_to_matrix(axis_angle)
poses_translation = torch.zeros((N_images, 2))

poses_py = poses.detach().cpu().numpy()
poses_translation_py = poses_translation.detach().cpu().numpy()


print("Min translation", torch.min(poses_translation))
print("Max translation", torch.max(poses_translation))

np.save(f"{folder_experiment}poses.npy", poses_py)
np.save(f"{folder_experiment}poses_translation.npy", poses_translation_py)
torch.save(poses, f"{folder_experiment}poses")
torch.save(poses_translation, f"{folder_experiment}poses_translation")

 
center_vector = utils.compute_center_of_mass(centering_structure)

all_images = []
with warnings.catch_warnings():
    warnings.simplefilter('ignore', BiopythonWarning)
    for i, structure in tqdm(enumerate(sorted_structures)):
        posed_structure = utils.compute_poses(structure, poses_py[i], poses_translation_py[i], center_vector)
        backbone = utils.get_backbone(parser.get_structure("A", struct))[None, :, :]
        backbones = torch.concatenate([backbone for _ in range(N_pose_per_struct)], dim=0)
        batch_images = renderer.compute_x_y_values_all_atoms(batch_structures, poses, poses_translation)
        all_images.append(batch_images)


all_images = torch.concatenate(all_images, dim=0)
torch.save(all_images, f"{folder_experiment}ImageDataSetNoNoise")
mean_variance_signal = torch.mean(torch.var(all_images, dim=(-2, -1)))
noise_var = mean_variance_signal*10
print(f"Mean variance of non noisy images: {mean_variance_signal}, adding noise with variance {noise_var}.")
all_images += torch.randn((N_images, N_pix, N_pix), device=device)*np.sqrt(noise_var)
torch.save(all_images, f"{folder_experiment}ImageDataSet")
all_images_np = np.transpose(all_images.detach().cpu().numpy(), axes=(0, 2, 1))
mrc.write(f"{folder_experiment}ImageDataSet.mrcs", all_images.detach().cpu().numpy(), Apix=Apix, is_vol=False)
with open(f"{folder_experiment}poses.pkl", "wb") as f:
	pickle.dump((poses_py, poses_translation_py), f)

torch.save(all_images[1:experiment_settings["N_pose_per_structure"]*10:experiment_settings["N_pose_per_structure"]], f"{folder_experiment}ExcerptImageDataSetNoNoise")
mrc.write(f"{folder_experiment}ExcerptImageDataSet.mrcs", all_images.detach().cpu().numpy()[1:experiment_settings["N_pose_per_structure"]*10:experiment_settings["N_pose_per_structure"]], Apix=Apix, is_vol=False)



