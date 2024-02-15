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
        ## I changed for the mean
        print("LOSS has been changed for the mean over the pixels !!")
        return torch.mean(0.5*torch.mean((predicted_images - images)**2, dim=-1))
    else:
        rmsd_per_latent_per_batch = 0.5 * torch.sum((predicted_images - images[:, None, :]) ** 2, dim=-1)
        latent_distribution = torch.softmax(log_latent_distribution, dim=-1)
        return torch.mean(torch.sum(rmsd_per_latent_per_batch*latent_distribution, dim=-1))


def compute_loss(predicted_images, images, tracking_dict):
    """
    Compute the entire loss
    :param predicted_images: torch.tensor(batch_size, N_pix), predicted images
    :param images: torch.tensor(batch_size, N_pix), images
    :return:
    """
    rmsd = compute_rmsd(predicted_images, images)
    tracking_dict["rmsd"].append(rmsd.detach().cpu().numpy())
    loss = rmsd

    return loss
