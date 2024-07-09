import torch
import numpy as np
from model.renderer import primal_to_fourier2d, fourier2d_to_primal 



def calc_cor_loss(pred_images, gt_images, mask=None):
    """
    Compute the cross-correlation for each pair (predicted_image, true) image in a batch. And average them
    pred_images: torch.tensor(batch_size, side_shape**2) predicted images
    gt_images: torch.tensor(batch_size, side_shape**2) of true images, translated according to the poses.
    """
    if mask is not None:
        pred_images = mask(pred_images)
        gt_images = mask(gt_images)
        pixel_num = mask.num_masked
    else:
        pixel_num = pred_images.shape[-2] * pred_images.shape[-1]

    pred_images = torch.flatten(pred_images, start_dim=-2, end_dim=-1)
    gt_images = torch.flatten(gt_images, start_dim=-2, end_dim=-1)
    # b, h, w -> b, num_pix
    #pred_images = pred_images.flatten(start_dim=2)
    #gt_images = gt_images.flatten(start_dim=2)

    # b 
    dots = (pred_images * gt_images).sum(-1)
    # b -> b 
    err = -dots / (gt_images.std(-1) + 1e-5) / (pred_images.std(-1) + 1e-5)
    # b -> 1 value
    err = err.mean() / pixel_num
    return err


def compute_rmsd(predicted_images, images):
    """
    Computes the negative log Gaussian likelihood
    :param predicted_images: torch.tensor(N_batch, N_pix), images predicted by the network OR torch.tensor(latent_dim, N_pix)
    :param images: torch.tensor(N_batch, N_pix), true images
    :param latent_distribution torch.tensor(N_batch, latent_dim) log_probabilities of the latent variable values if
            the latent variable is categorical. Otherwise None.
    :return: torch.float32, average of rmsd over images
    """
    predicted_images = torch.flatten(predicted_images, start_dim=-2, end_dim=-1)
    images = torch.flatten(images, start_dim=-2, end_dim=-1)
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


def compute_continuity_loss(predicted_structures, true_structure, device):
    """
    predicted_structures: tensor(N_batch, N_atoms, 3) predicted structure
    true_structure: Polymer object
    """
    chain_ids = np.array(true_structure.chain_id)
    keep_dist = chain_ids[1:] == chain_ids[:-1]
    true_coord = torch.tensor(true_structure.coord, dtype=torch.float32, device=device)
    chain_pred_distances = torch.sum((predicted_structures[:, 1:] - predicted_structures[:, :-1])**2, dim=-1)[:, keep_dist]
    chain_true_distances = torch.sum((true_coord[1:] - true_coord[:-1])**2, dim=-1)[keep_dist]
    loss = torch.mean(torch.sum((chain_pred_distances - chain_true_distances[None, :])**2, dim=-1)/(len(chain_ids)-1))
    return loss


def compute_clashing_distances(new_structures, device):
    """
    Computes the clashing distance loss. The cutoff is set to 4Å for non contiguous residues and the distance above this cutoff
    are not penalized
    Computes the distances between all the atoms
    :param new_structures: torch.tensor(N_batch, N_residues, 3), atom positions
    :return: torch.tensor(1, ) of the averaged clashing distance for distance inferior to 4Å,
    reaverage over the batch dimension
    """
    N_residues = new_structures.shape[1]
    #distances is torch.tensor(N_batch, N_residues, N_residues)
    distances = torch.cdist(new_structures, new_structures)
    triu_indices = torch.triu_indices(N_residues, N_residues, offset=2, device=device)
    distances = distances[:, triu_indices[0], triu_indices[1]]
    number_clash_per_sample = torch.sum(distances < 4, dim=-1)
    distances = torch.minimum((distances - 4), torch.zeros_like(distances))**2
    average_clahing = torch.sum(distances, dim=-1)/number_clash_per_sample
    return torch.mean(average_clahing)


def compute_loss(predicted_images, images, mask_image, latent_mean, latent_std, vae, loss_weights,
                 experiment_settings, tracking_dict, predicted_structures = None, true_structure=None, device=None):
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
    rmsd = calc_cor_loss(predicted_images, images, mask_image)
    KL_prior_latent = compute_KL_prior_latent(latent_mean, latent_std, experiment_settings["epsilon_kl"])
    l2_pen = compute_l2_pen(vae)
    continuity_loss = compute_continuity_loss(predicted_structures, true_structure, device)
    clashing_loss = compute_clashing_distances(predicted_structures, device)


    tracking_dict["rmsd"].append(rmsd.detach().cpu().numpy())
    tracking_dict["kl_prior_latent"].append(KL_prior_latent.detach().cpu().numpy())
    tracking_dict["l2_pen"].append(l2_pen.detach().cpu().numpy())
    tracking_dict["continuity_loss"].append(continuity_loss.detach().cpu().numpy())
    tracking_dict["clashing_loss"].append(clashing_loss.detach().cpu().numpy())

    loss = rmsd + loss_weights["KL_prior_latent"]*KL_prior_latent \
           + loss_weights["l2_pen"] * l2_pen + loss_weights["clashing_loss"]*clashing_loss\
           + loss_weights["continuity_loss"]*continuity_loss


    return loss
