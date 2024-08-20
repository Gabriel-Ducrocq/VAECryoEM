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
import einops

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
    vae, image_translator, ctf, grid, gmm_repr, optimizer, dataset, N_epochs, batch_size, experiment_settings, latent_type, device, scheduler, base_structure, lp_mask2d, mask_images, amortized = model.utils.parse_yaml(yaml_setting_path)
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

    for epoch in range(N_epochs):
        print("Epoch number:", epoch)
        tracking_metrics = {"rmsd":[], "kl_prior_latent":[], "kl_prior_mask_mean":[], "kl_prior_mask_std":[],
                            "kl_prior_mask_proportions":[], "l2_pen":[], "continuity_loss":[], "clashing_loss":[]}

        #### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DROP LAST !!!!!! ##################################
        data_loader = tqdm(iter(DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers = 4, drop_last=True)))
        start_tot = time()
        for batch_num, (indexes, batch_images, batch_poses, batch_poses_translation) in enumerate(data_loader):
            #start = time()
            batch_images = batch_images.to(device)
            batch_poses = batch_poses.to(device)
            batch_poses_translation = batch_poses_translation.to(device)
            indexes = indexes.to(device)
            #plt.imshow(batch_images[0].detach().cpu())
            #plt.show()
            #start_net = time()
            flattened_batch_images = batch_images.flatten(start_dim=-2)
            batch_translated_images = image_translator.transform(batch_images, batch_poses_translation[:, None, :])
            lp_batch_translated_images = low_pass_images(batch_translated_images, lp_mask2d)
            if not amortized:
                latent_variables, latent_mean, latent_std = vae.sample_latent(flattened_batch_images)
            else:
                latent_variables, latent_mean, latent_std = vae.sample_latent(None, indexes)

            mask = vae.sample_mask(batch_images.shape[0])
            quaternions_per_domain, translations_per_domain = vae.decode(latent_variables)
            #start_old = time()
            #rotation_per_residue = model.utils.compute_rotations_per_residue(quaternions_per_domain, mask, device)
            #end_old = time()
            #start_new = time()
            #rotation_per_residue = model.utils.compute_rotations_per_residue_einops(quaternions_per_domain, mask, device)
            #rotation_per_residue = model.utils.compute_rotations_per_residue(quaternions_per_domain, mask, device)
            #end_new = time()
            #print("Old and New", end_old - start_old, end_new - start_new)
            #print("\n\n\n")
            #print("ARE THE TWO ROT EQUAL", torch.allclose(rotation_per_residue,rotation_per_residue_einops))
            #print("\n\n\n")
            translation_per_residue = model.utils.compute_translations_per_residue(translations_per_domain, mask)
            #end_net = time()
            #print("Net time:", end_net - start_net)
            #start_deforming = time()
            #predicted_structures = model.utils.deform_structure(gmm_repr.mus, translation_per_residue,
            #                                                   rotation_per_residue)

            predicted_structures = model.utils.deform_structure_bis(gmm_repr.mus, translation_per_residue, quaternions_per_domain, mask, device)
            posed_predicted_structures = renderer.rotate_structure(predicted_structures, batch_poses)
            #posed_predicted_structures = predicted_structures


            #end_deforming = time()
            #print("Deforming time", end_deforming - start_deforming)
            #start_proj = time()
            #start_real = time()
            predicted_images  = renderer.project(posed_predicted_structures, gmm_repr.sigmas, gmm_repr.amplitudes, grid)
            #end_real = time()
            #print("Time real", end_real - start_real)
            #print("Percentage of freqs kept:", np.mean((torch.sqrt(torch.sum(ctf.freqs**2, dim=1)) < 1/4.0).detach().cpu().numpy()))
            #print("Number of freqs kept:", np.sum((torch.sqrt(torch.sum(ctf.freqs**2, dim=1)) < 1/4.0).detach().cpu().numpy()))
            #freqs_proj = torch.fft.fftshift(torch.fft.fftfreq(128, dataset.down_apix, device=device))[:2]
            #print("SHAPE", freqs_proj.shape)

            #start_fourier = time()
            #predicted_images_fourier = renderer.project_fourier(posed_predicted_structures, gmm_repr.sigmas, gmm_repr.amplitudes, freqs_proj)
            #end_fourier = time()
            #print("Time fourier", end_fourier - start_fourier)

            #pred_images_ifft = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(predicted_images_fourier))).real

            #plt.imshow(torch.fft.ifftshift(torch.fft.fft2(torch.fft.fftshift(predicted_images)))[0].detach().numpy().real)
            #plt.show()
            #print(pred_images_ifft[0].detach().numpy())
            #plt.imshow(predicted_images_fourier[0].detach().numpy().real)
            #plt.show()


            #plt.imshow(predicted_images[0].detach().numpy())
            #plt.show()
            #print(pred_images_ifft[0].detach().numpy())
            #plt.imshow(pred_images_ifft[0].detach().numpy().real)
            #plt.show()





            #proj_base = renderer.project(gmm_repr.mus[None, :, :], gmm_repr.sigmas, gmm_repr.amplitudes, grid)
            #plt.imshow(proj_base[0].detach().cpu())
            #plt.savefig("true_image.png")
            #end_proj = time()
            #print("Proj time", end_proj- start_proj)
            #start_ctf = time()


            batch_predicted_images = renderer.apply_ctf(predicted_images, ctf, indexes)
            #batch_predicted_images = renderer.apply_ctf(pred_images_ifft, ctf, indexes)
            print("\n\n")



            #batch_predicted_images = predicted_images
            #end_ctf = time()
            #print("CTF time", end_ctf - start_ctf)
            #start_trans = time()
            #end_trans = time()
            #print("Tran time", end_trans- start_trans)
            #start_flatten = time()
            #batch_predicted_images = torch.flatten(batch_predicted_images, start_dim=-2, end_dim=-1)
            #end_flatten = time()
            #print("FLATTEN time", end_flatten - start_flatten)
            #batch_predicted_images = dataset.standardize(batch_predicted_images, device=device)

            #print("True images mean", torch.mean(batch_images), "True images std", torch.std(batch_images))
            #print("Pred images mean", torch.mean(batch_predicted_images), "Pred images std", torch.std(batch_predicted_images))
            #start_loss = time()
            #mask_images
            loss = compute_loss(batch_predicted_images, lp_batch_translated_images, None, latent_mean, latent_std, vae,
                                experiment_settings["loss_weights"], experiment_settings, tracking_metrics, segments=mask, predicted_structures=predicted_structures, true_structure=base_structure, device=device)

            #end_loss = time()
            #print("Loss time", end_loss - start_loss)
            #print("Epoch:",  epoch, "Batch number:", batch_num, "Loss:", loss, "device:", device)
            #start_gradient = time()
            #start_backward = time()
            loss.backward()
            for n_chain in vae.chain_ids:
                print(vae.dict_segments_means_means[f"chain_{n_chain}"].grad)
                print("\n")
                print(vae.dict_segments_means_stds[f"chain_{n_chain}"].grad)
                print("\n")
                print(vae.dict_segments_stds_means[f"chain_{n_chain}"].grad)
                print("\n")
                print(vae.dict_segments_stds_stds[f"chain_{n_chain}"].grad)
                print("\n")
                print(vae.dict_segments_proportions_means[f"chain_{n_chain}"].grad)
                print("\n")
                print(vae.dict_segments_proportions_stds[f"chain_{n_chain}"].grad)
            

            print("\n\n\n\n\n\n")

            optimizer.step()
            #end_backward = time()
            #print("TIME BACKWARD", end_backward - start_backward)
            optimizer.zero_grad()
            #end_gradient = time()
            #print("Gradient time", end_gradient - start_gradient)
            #end = time()
            #print("Iteration duration:", end-start)


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

