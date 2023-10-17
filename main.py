import torch
import numpy as np
import model.utils
from model.vae import VAE
from model.mlp import MLP
from model.renderer import Renderer
from model.dataset import ImageDataSet
from torch.utils.data import DataLoader
from model.loss import compute_loss

def train(yaml_setting_path):
    """
    train a VAE network
    :param yaml_setting_path: str, path the yaml containing all the details of the experiment
    :return:
    """
    experiment_settings = model.utils.parse_yaml(yaml_setting_path)
    image_settings = model.utils.parse_yaml(experiment_settings["image_yaml"])
    if experiment_settings["device"] == "GPU":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"


    for mask_prior_key in experiment_settings["mask_prior"].keys():
        experiment_settings["mask_prior"][mask_prior_key]["mean"] = torch.tensor(experiment_settings["mask_prior"][mask_prior_key]["mean"],
                                                                                 dtype=torch.float32, device=device)
        experiment_settings["mask_prior"][mask_prior_key]["std"] = torch.tensor(experiment_settings["mask_prior"][mask_prior_key]["std"],
                                                                                 dtype=torch.float32, device=device)

    encoder = MLP(image_settings["N_pixels_per_axis"][0]*image_settings["N_pixels_per_axis"][1], experiment_settings["latent_dimension"]*2,
                  experiment_settings["encoder"]["hidden_dimensions"], network_type="encoder", device=device)
    decoder = MLP(experiment_settings["latent_dimension"], experiment_settings["N_domains"]*6,
                  experiment_settings["decoder"]["hidden_dimensions"], network_type="decoder", device=device)

    vae = VAE(encoder, decoder, device, N_domains = experiment_settings["N_domains"], N_residues= experiment_settings["N_residues"],
              tau_mask=experiment_settings["tau_mask"], mask_start_values=experiment_settings["mask_start"])

    pixels_x = np.linspace(image_settings["image_lower_bounds"][0], image_settings["image_upper_bounds"][0],
                           num=image_settings["N_pixels_per_axis"][0]).reshape(1, -1)

    pixels_y = np.linspace(image_settings["image_lower_bounds"][1], image_settings["image_upper_bounds"][1],
                           num=image_settings["N_pixels_per_axis"][1]).reshape(1, -1)

    renderer = Renderer(pixels_x, pixels_y, N_atoms = experiment_settings["N_residues"]*3,
                        period = image_settings["renderer"]["period"], std = 1, defocus = image_settings["renderer"]["defocus"],
                        spherical_aberration = image_settings["renderer"]["spherical_aberration"],
                        accelerating_voltage = image_settings["renderer"]["accelerating_voltage"],
                        amplitude_contrast_ratio = image_settings["renderer"]["amplitude_contrast_ratio"],
                        device = device, use_ctf = image_settings["renderer"]["use_ctf"])

    base_structure = model.utils.read_pdb(experiment_settings["base_structure_path"])
    atom_positions_non_centered = model.utils.get_backbone(base_structure)
    atom_positions = atom_positions_non_centered - np.mean(atom_positions_non_centered, axis=0)
    atom_positions = torch.tensor(atom_positions, dtype=torch.float32)

    if experiment_settings["optimizer"]["name"] == "adam":
        optimizer = torch.optim.Adam(vae.parameters(), lr=experiment_settings["optimizer"]["learning_rate"])
    else:
        raise Exception("Optimizer must be Adam")

    dataset = ImageDataSet(experiment_settings["dataset_images_path"], experiment_settings["dataset_poses_path"])
    N_epochs = experiment_settings["N_epochs"]
    batch_size = experiment_settings["batch_size"]
    vae.to(device)
    for epoch in range(N_epochs):
        data_loader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=True))
        for batch_images, batch_poses in data_loader:
            batch_images.to(device)
            batch_poses.to(device)
            latent_variables, latent_mean, latent_std = vae.sample_latent(batch_images)
            mask = vae.sample_mask()
            quaternions_per_domain, translations_per_domain = vae.decode(latent_variables)
            rotation_per_residue = model.utils.compute_rotations_per_residue(quaternions_per_domain, mask, device)
            translation_per_residue = model.utils.compute_translations_per_residue(translations_per_domain, mask)
            deformed_structures = model.utils.deform_structure(atom_positions, translation_per_residue,
                                                               rotation_per_residue)

            batch_predicted_images = renderer.compute_x_y_values_all_atoms(deformed_structures, batch_poses)
            batch_predicted_images = torch.flatten(batch_predicted_images, start_dim=-2, end_dim=-1)
            loss, rmsd = compute_loss(batch_predicted_images, batch_images, latent_mean, latent_std, vae,
                                    experiment_settings["loss_weights"], experiment_settings)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(loss, rmsd)








# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train("data/debug_run/parameters.yaml")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
