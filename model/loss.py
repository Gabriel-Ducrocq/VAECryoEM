import torch

def compute_rmsd(predicted_images, images):
    """
    Computes the negative log Gaussian likelihood
    :param predicted_images: torch.tensor(N_batch, N_pix), images predicted by the network
    :param images: torch.tensor(N_batch, N_pix), true images
    :return: torch.float32, average of rmsd over images
    """
    return torch.mean(0.5*torch.sum((predicted_images - images)**2, dim=-1))

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


def compute_loss(predicted_images, images, latent_mean, latent_std, vae, loss_weights,
                 experiment_settings):
    """
    Compute the entire loss
    :param predicted_images: torch.tensor(batch_size, N_pix), predicted images
    :param images: torch.tensor(batch_size, N_pix), images
    :param latent_mean:torch.tensor(batch_size, latent_dim), mean of the approximate latent distribution
    :param latent_std:torch.tensor(batch_size, latent_dim), std of the approximate latent distribution
    :param mask_prior: dict containing the tensors of the parameters of the prior distribution
    :param vae: torch.nn.Module
    :param loss_weights: dict containing the weights of each part of the losses
    :param epsilon_loss: float, epsilon added in the log to avoid log(0)
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
    loss = rmsd + loss_weights["KL_prior_latent"]*KL_prior_latent \
           + loss_weights["KL_prior_mask_mean"]*KL_prior_mask_means \
           + loss_weights["KL_prior_mask_std"] * KL_prior_mask_stds \
           + loss_weights["KL_prior_mask_proportions"] * KL_prior_mask_proportions \
           + loss_weights["l2_pen"] * l2_pen

    return loss, rmsd
