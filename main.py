import torch
import wandb
import argparse
import model.utils
import numpy as np
from tqdm import tqdm
from time import time
from model import renderer
from model.loss import compute_loss
from model.utils import low_pass_images
from torch.utils.data import DataLoader
from pytorch3d.transforms import rotation_6d_to_matrix

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
    vae, image_translator, ctf, grid, gmm_repr, optimizer, dataset, N_epochs, batch_size, experiment_settings, latent_type, device, scheduler, _, lp_mask2d, mask_images = model.utils.parse_yaml(yaml_setting_path)
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

    data_loader = tqdm(iter(DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers = 4, drop_last=True)))
    for epoch in range(5):
        all_losses = []
        for batch_num, (indexes, batch_images, batch_poses, batch_poses_translation) in enumerate(data_loader):
            batch_images = batch_images.to(device)
            batch_poses = batch_poses.to(device)
            batch_poses_translation = batch_poses_translation.to(device)
            indexes = indexes.to(device)
            flattened_batch_images = batch_images.flatten(start_dim=-2)
            ###### !!!!!!!!!!!!!!!!!!!!          BE CAREFUL FOR NOW I AM ONLY PREDICTING ROTATIONS POSE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   ######################
            batch_translated_images = image_translator.transform(batch_images, batch_poses_translation[:, None, :])
            lp_batch_translated_images = low_pass_images(batch_translated_images, lp_mask2d)
            latent_variables, predicted_rotation_pose, latent_mean, latent_std = vae.sample_latent(flattened_batch_images)
            target = torch.zeros_like(predicted_rotation_pose, dtype=torch.float32, device=device)
            target[:, 0] = 1
            target[:, 3] = 1
            loss = torch.mean(torch.sum((predicted_rotation_pose - target)**2, dim=-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            all_losses.append(loss.detach().cpu().numpy())

        wandb.log({"warm_up_loss": np.mean(all_losses)})


    for epoch in range(N_epochs):
        print("Epoch number:", epoch)
        tracking_metrics = {"rmsd":[], "kl_prior_latent":[], "kl_prior_mask_mean":[], "kl_prior_mask_std":[],
                            "kl_prior_mask_proportions":[], "l2_pen":[], "viewpoint_angle_diff_degrees":[]}

        #### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DROP LAST !!!!!! ##################################
        data_loader = tqdm(iter(DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers = 4, drop_last=True)))
        start_tot = time()
        for batch_num, (indexes, batch_images, batch_poses, batch_poses_translation) in enumerate(data_loader):
            start_batch = time()
            #start = time()
            batch_images = batch_images.to(device)
            batch_poses = batch_poses.to(device)
            batch_poses_translation = batch_poses_translation.to(device)
            indexes = indexes.to(device)
            #plt.imshow(batch_images[0].detach().cpu())
            #plt.show()
            #start_net = time()
            flattened_batch_images = batch_images.flatten(start_dim=-2)
            ###### !!!!!!!!!!!!!!!!!!!!          BE CAREFUL FOR NOW I AM ONLY PREDICTING ROTATIONS POSE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   ######################
            batch_translated_images = image_translator.transform(batch_images, batch_poses_translation[:, None, :])
            lp_batch_translated_images = low_pass_images(batch_translated_images, lp_mask2d)
            latent_variables, predicted_rotation_pose, latent_mean, latent_std = vae.sample_latent(flattened_batch_images)
            predicted_rotation_matrix_pose = rotation_6d_to_matrix(predicted_rotation_pose)
            #mask = vae.sample_mask(batch_images.shape[0])
            #if epoch < 30:
            #    latent_variables = torch.randn_like(latent_variables, device=device, dtype=torch.float32)

            #quaternions_per_domain, translations_per_domain = vae.decode(latent_variables)
            #translation_per_residue = model.utils.compute_translations_per_residue(translations_per_domain, mask)
            #predicted_structures = model.utils.deform_structure_bis(gmm_repr.mus, translation_per_residue, quaternions_per_domain, mask, device)
            predicted_structures = gmm_repr.mus[None, :, :].repeat(batch_size, 1, 1)
            posed_predicted_structures = renderer.rotate_structure(predicted_structures, predicted_rotation_matrix_pose)
            predicted_images = renderer.project(posed_predicted_structures, gmm_repr.sigmas, gmm_repr.amplitudes, grid)
            print("Time to projection:", end-start)
            batch_predicted_images = renderer.apply_ctf(predicted_images, ctf, indexes)

            if not experiment_settings["clashing_loss"]:
                deformed_structures = None
            loss = compute_loss(batch_predicted_images, lp_batch_translated_images, predicted_rotation_matrix_pose, batch_poses, None, latent_mean, latent_std, vae,
                                experiment_settings["loss_weights"], experiment_settings, tracking_metrics, device=device)
                                #predicted_structures=deformed_structures)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #end_batch = time()
            #print("time batch", end_batch - start_batch)


        end_tot = time()
        print("TOTAL TIME", end_tot - start_tot)

            

        if scheduler:
            scheduler.step()

        if not debug_mode:
            model.utils.monitor_training(mask, tracking_metrics, epoch, experiment_settings, vae, optimizer)


if __name__ == '__main__':
    wandb.login()

    args = parser_arg.parse_args()
    path = args.experiment_yaml
    debug_mode = args.debug
    from torch import autograd
    with autograd.detect_anomaly():
        train(path, debug_mode)

