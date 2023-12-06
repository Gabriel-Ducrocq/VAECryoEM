import torch
import numpy as np

def compute_rmsd(predicted_images, images, log_latent_distribution = None, type="continuous"):
    """
    Computes the negative log Gaussian likelihood
    :param predicted_images: torch.tensor(N_batch, N_pix), images predicted by the network OR torch.tensor(latent_dim, N_pix)
    :param images: torch.tensor(N_batch, N_pix), true images
    :param latent_distribution torch.tensor(N_batch, latent_dim) log_probabilities of the latent variable values if
            the latent variable is categorical. Otherwise None.
    :return: torch.float32, average of rmsd over images
    """
    assert type in ["continuous", "categorical"]
    if type == "continuous":
        return torch.mean(0.5*torch.sum((predicted_images - images)**2, dim=-1))
    else:
        rmsd_per_latent_per_batch = 0.5 * torch.sum((predicted_images - images[:, None, :]) ** 2, dim=-1)
        latent_distribution = torch.softmax(log_latent_distribution, dim=-1)
        return torch.mean(torch.sum(rmsd_per_latent_per_batch*latent_distribution, dim=-1))

def compute_KL_prior_latent_translation(latent_mean, latent_std, epsilon_loss):
    """
    Computes the KL divergence between the approximate posterior and the prior over the translations,
    where the latent prior is given by a standard Gaussian distribution.
    :param latent_mean: torch.tensor(N_batch, N_domains, 3), mean of the Gaussian for translation
    :param latent_std: torch.tensor(N_batch, N_domains, 3), std of the Gaussian for translation
    :param epsilon_loss: float, a constant added in the log to avoid log(0) situation.
    :return: torch.float32, average of the KL losses accross batch samples for translations
    """
    return torch.mean(torch.sum(-0.5 * torch.sum(1 + torch.log(latent_std ** 2 + eval(epsilon_loss)) \
                                           - latent_mean ** 2 \
                                           - latent_std ** 2, dim=-1)), dim=-1)


def log_gaussian_pdf(x, stds):
    """
    compute the log pdf of Gaussian with mean 0
    :param x: torch.tensor(N_batch, N_domains, N_k, 3)
    :param stds: torch.tensor(N_batch, N_domains, 3)
    :return: torch.tensor(N_batch, N_domains, N_k)
    """
    return -0.5* torch.sum(x**2/stds[:, :, None, :]**2, dim=-1) - 0.5*torch.sum(torch.log(stds), dim=-1)[:, :, None] - 1.5*np.log(2*np.pi)

def compute_entropy_rotations(std_rot, noise_rot, K=10):
    """
    Compute the entropy of the distribution over SO(3)
    :param std_rot: torch.tensor(N_batch, N_domains, 3) of standard deviation over R^3
    :param noise_rot: torch.tensor(N_batch, N_domains, 3) of sampled noise
    :return:
    """
    twokpi = torch.linspace(-K, K, steps=2*K)[None, None, :]
    norms = torch.sqrt(torch.sum(noise_rot**2, dim=-1))[:, :, None]
    normalized_noise_rot = noise_rot/norms
    norms_plus_twokpi = norms + twokpi
    first_term = log_gaussian_pdf(torch.einsum("bdk, bdl -> bdkl", norms_plus_twokpi, normalized_noise_rot), std_rot)
    second_term = torch.log(norms_plus_twokpi**2/(2-2*torch.cos(norms)))
    #The tensor first_plus_second is (N_batch, N_domains, N_k)
    first_plus_second = first_term + second_term
    entropy_per_domain = torch.logsumexp(first_plus_second, dim=-1)
    entropy = torch.mean(torch.sum(entropy_per_domain, dim=-1))
    return entropy





def compute_KL_prior_latent_discrete(log_latent_distribution):
    """
    Computes the KL divergence between the approximate posterior and the prior over the latent,
    where the latent prior is given by a standard Gaussian distribution.
    :param log_latent_distrib: torch.tensor(N_batch, latent_dim), log of the probabilities for the values of the latent
    :return: torch.float32, average of the KL losses accross batch samples
    """
    latent_dim = log_latent_distribution.shape[-1]
    latent_distribution = torch.softmax(log_latent_distribution, dim=-1)
    return torch.mean(-torch.sum((log_latent_distribution + np.log(latent_dim))*latent_distribution, dim=-1))





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


def compute_loss(predicted_images, images, translation_mean, translation_std, std_rot, noise_rot, vae, loss_weights,
                 experiment_settings, tracking_dict, log_latent_distribution=None, type="continuous"):
    """
    Compute the entire loss
    :param predicted_images: torch.tensor(batch_size, N_pix), predicted images
    :param images: torch.tensor(batch_size, N_pix), images
    :param translation_mean: torch.tensor(batch_size, N_domains, 3) mean of the distributions over the translations
    :param translation_std: torch.tensor(batch_size, N_domains, 3) std of thedistribution over the translations
    :param std_rot: torch.tensor(N_batch, N_domains, 3) of standard deviation over R^3
    :param noise_rot: torch.tensor(N_batch, N_domains, 3) of sampled noise
    :param mask_prior: dict containing the tensors of the parameters of the prior distribution
    :param vae: torch.nn.Module
    :param loss_weights: dict containing the weights of each part of the losses
    :param epsilon_loss: float, epsilon added in the log to avoid log(0)
    :param log_latent_distribution: torch.tensor(N_batch. latent_dim) if latent is categorical, else None
    :return:
    """
    assert type in ["continuous", "categorical"]
    rmsd = compute_rmsd(predicted_images, images)
    KL_prior_translations = compute_KL_prior_latent_translation(translation_mean, translation_std, experiment_settings["epsilon_kl"])
    KL_prior_rotations = compute_entropy_rotations(std_rot, noise_rot)

    KL_prior_mask_means = compute_KL_prior_mask(
        vae.mask_parameters, experiment_settings["mask_prior"],"means", epsilon_kl=experiment_settings["epsilon_kl"])
    KL_prior_mask_stds = compute_KL_prior_mask(vae.mask_parameters, experiment_settings["mask_prior"],
                                               "stds", epsilon_kl=experiment_settings["epsilon_kl"])
    KL_prior_mask_proportions = compute_KL_prior_mask(vae.mask_parameters, experiment_settings["mask_prior"],
                                               "proportions", epsilon_kl=experiment_settings["epsilon_kl"])
    l2_pen = compute_l2_pen(vae)

    tracking_dict["rmsd"].append(rmsd.detach().cpu().numpy())
    tracking_dict["kl_prior_translation"].append(KL_prior_translations.detach().cpu().numpy())
    tracking_dict["kl_prior_rotation"].append(KL_prior_rotations.detach().cpu().numpy())
    tracking_dict["kl_prior_mask_mean"].append(KL_prior_mask_means.detach().cpu().numpy())
    tracking_dict["kl_prior_mask_std"].append(KL_prior_mask_stds.detach().cpu().numpy())
    tracking_dict["kl_prior_mask_proportions"].append(KL_prior_mask_proportions.detach().cpu().numpy())
    tracking_dict["l2_pen"].append(l2_pen.detach().cpu().numpy())

    loss = rmsd + loss_weights["KL_prior_latent"]*KL_prior_translations \
           + loss_weights["KL_prior_latent"]*KL_prior_rotations \
           + loss_weights["KL_prior_mask_mean"]*KL_prior_mask_means \
           + loss_weights["KL_prior_mask_std"] * KL_prior_mask_stds \
           + loss_weights["KL_prior_mask_proportions"] * KL_prior_mask_proportions \
           + loss_weights["l2_pen"] * l2_pen

    return loss
