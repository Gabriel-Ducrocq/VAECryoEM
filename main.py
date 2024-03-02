import torch
import wandb
import argparse
import model.utils
from time import time
from model import renderer
from model.loss import compute_loss
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--experiment_yaml', type=str, required=True)
parser_arg.add_argument('--debug', type=bool, required=False)


def train(yaml_setting_path, debug_mode):
    """
    train a VAE network
    :param yaml_setting_path: str, path the yaml containing all the details of the experiment
    :return:
    """
    vae, ctf, grid, gmm_repr, optimizer, dataset, N_epochs, batch_size, experiment_settings, latent_type, device, scheduler = model.utils.parse_yaml(yaml_setting_path)
    if experiment_settings["resume_training"]["model"] != "None":
        name = f"experiment_{experiment_settings['name']}_resume"
    else:
        name = f"experiment_{experiment_settings['name']}"

    if not debug_mode:
        wandb.init(
            # Set the project where this run will be logged
            project="VAECryoEM",
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
                name=name,


            # Track hyperparameters and run metadata
            config={
                "learning_rate": experiment_settings["optimizer"]["learning_rate"],
                "architecture": "VAE",
                "dataset": experiment_settings["star_file"],
                "epochs": experiment_settings["N_epochs"],
            })

    #non_noisy_images = torch.load("data/dataset/spike/ImageDataSetNoNoiseNoCTF")
    for epoch in range(N_epochs):
        print("Epoch number:", epoch)
        tracking_metrics = {"rmsd":[], "kl_prior_latent":[], "kl_prior_mask_mean":[], "kl_prior_mask_std":[],
                            "kl_prior_mask_proportions":[], "l2_pen":[]}

        data_loader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=True))
        for batch_num, (indexes, batch_images, batch_poses, batch_poses_translation) in enumerate(data_loader):
            start = time()
            batch_images = batch_images.to(device)
            batch_poses = batch_poses.to(device)
            batch_poses_translation = batch_poses_translation.to(device)
            indexes = indexes.to(device)
            latent_variables, latent_mean, latent_std = vae.sample_latent(batch_images)

            mask = vae.sample_mask(batch_size)
            quaternions_per_domain, translations_per_domain = vae.decode(latent_variables)
            rotation_per_residue = model.utils.compute_rotations_per_residue(quaternions_per_domain, mask, device)
            translation_per_residue = model.utils.compute_translations_per_residue(translations_per_domain, mask)
            predicted_structures = model.utils.deform_structure(gmm_repr.mus, translation_per_residue,
                                                               rotation_per_residue)

            posed_predicted_structures = renderer.get_posed_structure(predicted_structures, batch_poses, batch_poses_translation)

            predicted_images = renderer.project(posed_predicted_structures, gmm_repr.sigmas, gmm_repr.amplitudes, grid)
            #batch_predicted_images = renderer.apply_ctf(predicted_images, ctf, indexes)
            batch_predicted_images = predicted_images
            batch_predicted_images = torch.flatten(batch_predicted_images, start_dim=-2, end_dim=-1)
            batch_predicted_images = dataset.standardize(batch_predicted_images, device=device)


            if not experiment_settings["clashing_loss"]:
                deformed_structures = None

            print("True images mean", torch.mean(batch_images), "True images std", torch.std(batch_images))
            print("Pred images mean", torch.mean(batch_predicted_images), "Pred images std", torch.std(batch_predicted_images))
            loss = compute_loss(batch_predicted_images, batch_images, latent_mean, latent_std, vae,
                                experiment_settings["loss_weights"], experiment_settings, tracking_metrics,
                                predicted_structures=deformed_structures)
            print("Epoch:",  epoch, "Batch number:", batch_num, "Loss:", loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            end = time()
            print("Iteration duration:", end-start)

        if scheduler:
            scheduler.step()

        if not debug_mode:
            model.utils.monitor_training(mask, tracking_metrics, epoch, experiment_settings, vae, optimizer)


if __name__ == '__main__':
    wandb.login()

    args = parser_arg.parse_args()
    path = args.experiment_yaml
    debug_mode = args.debug
    train(path, debug_mode)

