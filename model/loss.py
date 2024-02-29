import torch
import numpy as np

def compute_rmsd(predicted_images, images):
    """
    Computes the negative log Gaussian likelihood
    :param predicted_images: torch.tensor(N_batch, N_pix), images predicted by the network OR torch.tensor(latent_dim, N_pix)
    :param images: torch.tensor(N_batch, N_pix), true images
    :param latent_distribution torch.tensor(N_batch, latent_dim) log_probabilities of the latent variable values if
            the latent variable is categorical. Otherwise None.
    :return: torch.float32, average of rmsd over images
    """
    return torch.mean(0.5*torch.mean((predicted_images - images)**2, dim=-1))

def compute_KL_prior_latent(latent_mean, latent_std, epsilon_loss):
    """
    Computes the KL divergence between the approximate posterior and the prior over the latent,
    where the latent prior is given by a standard Gaussian distribution.
    :param latent_mean: torch.tensor(N_batch, latent_dim), mean of the Gaussian approximate posterior
    :param latent_std: torch.tensor(N_batch, latent_dim), std of the Gaussian approximate posterior
    :param epsilon_loss: float, a constant added in the log to avoid log(0) situation.
    :return: torch.float32, average of the KL losses accross batch samples
    """
    return torch.mean(-0.5 * torch.sum(1 + torch.log(latent_std ** 2 + eval(epsilon_loss)) \
                                           - latent_mean ** 2 \
                                           - latent_std ** 2, dim=1))


def compute_KL_prior_mask(mask_parameters, mask_prior, variable, epsilon_kl):
    """
    Compute the Dkl loss between the prior and the approximated posterior distribution
    :param mask_parameters: dictionnary, containing the tensor of mask parameters
    :param mask_prior: dictionnary, containing the tensor of mask prior
    :return: torch.float32,  Dkl loss
    """
    assert variable in ["means", "stds", "proportions"]
    return torch.sum(-1/2 + torch.log(mask_prior[variable]["std"]/mask_parameters[variable]["std"] + eval(epsilon_kl)) \
    + (1/2)*(mask_parameters[variable]["std"]**2 +
    (mask_prior[variable]["mean"] - mask_parameters[variable]["mean"])**2)/mask_prior[variable]["std"]**2)



def compute_l2_pen(network):
    """
    Compute the l2 norm of the network's weight
    :param network: torch.nn.Module
    :return: torch.float32, l2 squared norm of the network's weights
    """
    l2_pen = 0
    for name, p in network.named_parameters():
        if "weight" in name and ("encoder" in name or "decoder" in name):
            l2_pen += torch.sum(p ** 2)

    return l2_pen

def compute_clashing_distances(new_structures):
    """
    Computes the clashing distance loss. The cutoff is set to 4Å for non contiguous residues and the distance above this cutoff
    are not penalized
    Computes the distances between all C_\alpha atoms
    :param new_structures: torch.tensor(N_batch, 3*N_residues, 3), atom positions
    :return: torch.tensor(1, ) of the averaged clashing distance for distance inferior to 4Å,
    reaverage over the batch dimension
    """
    c_alphas = new_structures[:, ::3, :]
    N_residues = c_alphas.shape[1]
    #distances is torch.tensor(N_batch, N_residues, N_residues)
    distances = torch.cdist(c_alphas, c_alphas)
    triu_indices = torch.triu_indices(N_residues, N_residues, 1)
    distances = distances[:, triu_indices[0], triu_indices[1]]
    return torch.mean(torch.mean(torch.minimum(distances - 4, torch.zeros_like(distances))**2, dim=-1))


def compute_loss(predicted_images, images, latent_mean, latent_std, vae, loss_weights,
                 experiment_settings, tracking_dict, predicted_structures = None):
    """
    Compute the entire loss
    :param predicted_images: torch.tensor(batch_size, N_pix), predicted images
    :param images: torch.tensor(batch_size, N_pix), images
    :param latent_mean:torch.tensor(batch_size, latent_dim), mean of the approximate latent distribution
    :param latent_std:torch.tensor(batch_size, latent_dim), std of the approximate latent distribution
    :param mask_prior: dict containing the tensors of the parameters of the prior distribution
    :param vae: torch.nn.Module
    :param loss_weights: dict containing the weights of each part of the losses
    :param predicted_structures: torch.tensor(N_batch, 3*N_residues, 3)
    :return:
    """
    rmsd = compute_rmsd(predicted_images, images)
    KL_prior_latent = compute_KL_prior_latent(latent_mean, latent_std, experiment_settings["epsilon_kl"])
    KL_prior_mask_means = compute_KL_prior_mask(
        vae.mask_parameters, experiment_settings["mask_prior"],
        "means", epsilon_kl=experiment_settings["epsilon_kl"])
    KL_prior_mask_stds = compute_KL_prior_mask(vae.mask_parameters, experiment_settings["mask_prior"],
                                               "stds", epsilon_kl=experiment_settings["epsilon_kl"])
    KL_prior_mask_proportions = compute_KL_prior_mask(vae.mask_parameters, experiment_settings["mask_prior"],
                                               "proportions", epsilon_kl=experiment_settings["epsilon_kl"])
    l2_pen = compute_l2_pen(vae)
    clashing_loss = 0
    if predicted_structures is not None:
        clashing_loss = compute_clashing_distances(predicted_structures)


    tracking_dict["rmsd"].append(rmsd.detach().cpu().numpy())
    tracking_dict["kl_prior_latent"].append(KL_prior_latent.detach().cpu().numpy())
    tracking_dict["kl_prior_mask_mean"].append(KL_prior_mask_means.detach().cpu().numpy())
    tracking_dict["kl_prior_mask_std"].append(KL_prior_mask_stds.detach().cpu().numpy())
    tracking_dict["kl_prior_mask_proportions"].append(KL_prior_mask_proportions.detach().cpu().numpy())
    tracking_dict["l2_pen"].append(l2_pen.detach().cpu().numpy())

    loss = rmsd + loss_weights["KL_prior_latent"]*KL_prior_latent \
           + loss_weights["KL_prior_mask_mean"]*KL_prior_mask_means \
           + loss_weights["KL_prior_mask_std"] * KL_prior_mask_stds \
           + loss_weights["KL_prior_mask_proportions"] * KL_prior_mask_proportions \
           + loss_weights["l2_pen"] * l2_pen + loss_weights["clashing"]*clashing_loss

    return loss
