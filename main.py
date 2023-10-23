import torch
import wandb
import numpy as np
import model.utils
from time import time
from model.vae import VAE
from model.mlp import MLP
from model.loss import compute_loss
from model.renderer import Renderer
from torch.utils.data import DataLoader


def train(yaml_setting_path):
    """
    train a VAE network
    :param yaml_setting_path: str, path the yaml containing all the details of the experiment
    :return:
    """
    vae, renderer, atom_positions, optimizer, dataset, N_epochs, batch_size, experiment_settings, device = model.utils.parse_yaml(yaml_setting_path)
    wandb.init(
        # Set the project where this run will be logged
        project="VAECryoEM",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"experiment_{experiment_settings['name']}",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": experiment_settings["optimizer"]["learning_rate"],
            "architecture": "VAE",
            "dataset": experiment_settings["dataset_images_path"],
            "epochs": experiment_settings["N_epochs"],
        })

    #features = np.load("data/dataset/debug_run2/features.npy", allow_pickle=True)
    #features = features.item()
    #absolute_positions = torch.tensor(features["absolute_positions"] - np.mean(features["absolute_positions"], axis=0))
    #absolute_positions = torch.tensor(absolute_positions, dtype=torch.float32, device=device)
    #atom_positions = absolute_positions.to(device)
    #print("features", atom_positions.shape)

    for epoch in range(N_epochs):
        print("Epoch number:", epoch)
        tracking_metrics = {"rmsd":[], "kl_prior_latent":[], "kl_prior_mask_mean":[], "kl_prior_mask_std":[],
                            "kl_prior_mask_proportions":[], "l2_pen":[]}
        data_loader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=True))
        for batch_images, batch_poses in data_loader:
            start = time()
            print(batch_images.shape)
            batch_images = batch_images.to(device)
            batch_poses = batch_poses.to(device)
            latent_variables, latent_mean, latent_std = vae.sample_latent(batch_images)
            mask = vae.sample_mask()
            quaternions_per_domain, translations_per_domain = vae.decode(latent_variables)
            rotation_per_residue = model.utils.compute_rotations_per_residue(quaternions_per_domain, mask, device)
            translation_per_residue = model.utils.compute_translations_per_residue(translations_per_domain, mask)
            deformed_structures = model.utils.deform_structure(atom_positions, translation_per_residue,
                                                               rotation_per_residue)

            batch_predicted_images = renderer.compute_x_y_values_all_atoms(deformed_structures, batch_poses)
            batch_predicted_images = torch.flatten(batch_predicted_images, start_dim=-2, end_dim=-1)
            loss = compute_loss(batch_predicted_images, batch_images, latent_mean, latent_std, vae,
                                    experiment_settings["loss_weights"], experiment_settings, tracking_metrics)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            end = time()
            print("Iteration duration:", end-start)

        model.utils.monitor_training(mask, tracking_metrics, epoch, experiment_settings, vae)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    wandb.login()
    train("data/dataset/debug_run3/parameters.yaml")

