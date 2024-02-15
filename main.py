import torch
import wandb
import argparse
import model.utils
from time import time
from model.loss import compute_loss
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--experiment_yaml', type=str, required=True)

def train(yaml_setting_path):
    """
    train a VAE network
    :param yaml_setting_path: str, path the yaml containing all the details of the experiment
    :return:
    """
    vae, renderer, atom_positions, optimizer, dataset, N_epochs, batch_size, experiment_settings, latent_type, device, scheduler = model.utils.parse_yaml(yaml_setting_path)
    if experiment_settings["resume_training"]["model"] != "None":
        name = f"experiment_{experiment_settings['name']}_resume"
    else:
        name = f"experiment_{experiment_settings['name']}"

    wandb.init(
        # Set the project where this run will be logged
        project="VAECryoEM",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=name,


        # Track hyperparameters and run metadata
        config={
            "learning_rate": experiment_settings["optimizer"]["learning_rate"],
            "architecture": "VAE",
            "dataset": experiment_settings["dataset_images_path"],
            "epochs": experiment_settings["N_epochs"],
        })

    #non_noisy_images = torch.load("data/dataset/spike/ImageDataSetNoNoiseNoCTF")
    for epoch in range(N_epochs):
        print("Epoch number:", epoch)
        tracking_metrics = {"rmsd":[]}

        data_loader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=True))
        for batch_images, batch_poses, batch_poses_translation, indexes in data_loader:
            start = time()
            batch_images = batch_images.to(device)
            batch_poses = batch_poses.to(device)
            batch_poses_translation = batch_poses_translation.to(device)
            mask = vae.sample_mask(batch_size)
            indexes_py = indexes.cpu().detach().numpy()
            rotation_per_domain, translations_per_domain = vae.batch_transformations(indexes_py)
            #print(r6_per_domain)
            #print("\n\n")

            rotation_per_residue = model.utils.compute_rotations_per_residue(rotation_per_domain, mask, device, vae.representation)
            translation_per_residue = model.utils.compute_translations_per_residue(translations_per_domain, mask)
            deformed_structures = model.utils.deform_structure(atom_positions, translation_per_residue,
                                                               rotation_per_residue)

            batch_predicted_images = renderer.compute_x_y_values_all_atoms(deformed_structures, batch_poses,
                                                                batch_poses_translation,latent_type=latent_type)

            batch_predicted_images = torch.flatten(batch_predicted_images, start_dim=-2, end_dim=-1)

            loss = compute_loss(batch_predicted_images, batch_images, tracking_metrics)
            print("loss", loss)
            loss.backward()
            optimizer.step(noise=True)
            print(vae.translation_per_domain[0])
            #print("GRADIENTS")
            #for param in vae.parameters():
            #    if param.grad is not None:
            #        print(param)

            optimizer.zero_grad()
            end = time()
            print("Iteration duration:", end-start)

        if scheduler:
            scheduler.step()

        model.utils.monitor_training(mask, tracking_metrics, epoch, experiment_settings, vae, optimizer)


if __name__ == '__main__':
    wandb.login()

    args = parser_arg.parse_args()
    path = args.experiment_yaml
    train(path)

