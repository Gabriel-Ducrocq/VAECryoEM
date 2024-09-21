import torch
import numpy as np
from torch import linalg as LA
import torch.nn.functional as F
from scipy.spatial import distance
from model.renderer import primal_to_fourier2d, fourier2d_to_primal 



AA_ATOMS = ("CA", )
NT_ATOMS = ("C1'", )



def calc_clash_loss(pred_struc, pair_index, clash_cutoff=4.0):
    pred_dist = pred_struc[:, pair_index]  # bsz, num_pair, 2, 3
    pred_dist = LA.vector_norm(torch.diff(pred_dist, dim=-2), axis=-1).squeeze(-1)  # bsz, num_pair
    possible_clash_dist = pred_dist[pred_dist < clash_cutoff]
    if possible_clash_dist.numel() == 0:
        avg_loss = torch.tensor(0.0).to(pred_struc)
    else:
        possible_clash_loss = (clash_cutoff - possible_clash_dist)**2
        avg_loss = possible_clash_loss.mean()
    return avg_loss


def calc_pair_dist_loss(pred_struc, pair_index, target_dist, type="vanilla", chain_id=None):
    bsz = pred_struc.shape[0]
    pred_dist = pred_struc[:, pair_index]  # bsz, num_pair, 2, 3
    pred_dist = LA.vector_norm(torch.diff(pred_dist, dim=-2), axis=-1).squeeze(-1)  # bsz, num_pair
    if type == "vanilla":
        return F.mse_loss(pred_dist, target_dist.repeat(bsz, 1))


def calc_dist_by_pair_indices(coord_arr, pair_indices):
    coord_pair_arr = coord_arr[pair_indices]  # num_pair, 2, 3
    dist = np.linalg.norm(np.diff(coord_pair_arr, axis=1), ord=2, axis=-1)
    return dist.flatten()




def find_continuous_pairs(chain_id_arr, res_id_arr, atom_name_arr):
    pairs = []

    # res_id in different chains are duplicated, so loop on chains
    u_chain_id = np.unique(chain_id_arr)

    #For each chain
    for c_id in u_chain_id:
        #Identify which residue belong to that chain
        tmp_mask = chain_id_arr == c_id
        tmp_indices_in_pdb = np.nonzero(tmp_mask)[0]

        #Get these residues and their indexes
        tmp_res_id_arr = res_id_arr[tmp_mask]
        tmp_atom_name_arr = atom_name_arr[tmp_mask]

        # check is aa or nt
        tmp_atom_name_set = set(tmp_atom_name_arr)

        if len(tmp_atom_name_set.intersection(AA_ATOMS)) > len(tmp_atom_name_set.intersection(NT_ATOMS)):
            in_res_atom_names = AA_ATOMS
        elif len(tmp_atom_name_set.intersection(AA_ATOMS)) < len(tmp_atom_name_set.intersection(NT_ATOMS)):
            in_res_atom_names = NT_ATOMS
        else:
            raise NotImplemented("Cannot determine chain is amino acid or nucleotide.")

        # find pairs
        if len(in_res_atom_names) == 1:
            #Get the unique residue indices as well as their indexes array of that chain
            u_res_id, indices_in_chain = np.unique(tmp_res_id_arr, return_index=True)
            if len(u_res_id) != np.sum(tmp_mask):
                raise ValueError(f"Found duplicate residue id in single chain {c_id}.")

            #Pair each residue with the following one and stack their indices column wise
            indices_in_chain_pair = np.column_stack((indices_in_chain[:-1], indices_in_chain[1:]))

            # must be adjacent on residue id
            valid_mask = np.abs(np.diff(u_res_id[indices_in_chain_pair], axis=1)) == 1

            #Keep only the pairs that have actually subsequent indices in the chain.
            indices_in_chain_pair = indices_in_chain_pair[valid_mask.flatten()]

            #Get their inndexes in the entire PDB, not only within the chain
            indices_in_pdb_pair = tmp_indices_in_pdb[indices_in_chain_pair]
        elif len(in_res_atom_names) > 1:

            def _cmp(a, b):
                # res_id compare
                if a[0] != b[0]:
                    return a[0] - b[0]
                else:
                    # atom_name in the same order of AA_ATOMS or NT_ATOMS
                    return in_res_atom_names.index(a[1]) - in_res_atom_names.index(b[1])

            cache = list(zip(tmp_res_id_arr, tmp_atom_name_arr, tmp_indices_in_pdb))
            sorted_cache = list(sorted(cache, key=cmp_to_key(_cmp)))

            sorted_indices_in_pdb = [item[2] for item in sorted_cache]
            sorted_res_id = [item[0] for item in sorted_cache]

            indices_in_pdb_pair = np.column_stack((sorted_indices_in_pdb[:-1], sorted_indices_in_pdb[1:]))

            valid_mask = np.abs(np.diff(np.column_stack((sorted_res_id[:-1], sorted_res_id[1:])), axis=1)) <= 1

            indices_in_pdb_pair = indices_in_pdb_pair[valid_mask.flatten()]
        else:
            raise NotImplemented("No enough atoms to construct continuous pairs.")

        pairs.append(indices_in_pdb_pair)

    pairs = np.vstack(pairs)
    return pairs

def find_range_cutoff_pairs(coord_arr, min_cutoff=4., max_cutoff=10.):
    dist_map = distance.cdist(coord_arr, coord_arr, metric='euclidean')
    sel_mask = (dist_map <= max_cutoff) & (dist_map >= min_cutoff)
    indices_in_pdb = np.nonzero(sel_mask)
    indices_in_pdb = np.column_stack((indices_in_pdb[0], indices_in_pdb[1]))
    return indices_in_pdb


def remove_duplicate_pairs(pairs_a, pairs_b, remove_flip=True):
    """Remove pair b from a"""
    s = max(pairs_a.max(), pairs_b.max()) + 1
    # trick for fast comparison
    mask = np.zeros((s, s), dtype=bool)

    #np.ravel_multi_index gets the index of the elements in non linear shape as if the array was linear
    #so ravel_multi_index(pairs_a.T, mask.shape) get the indexes of the elements in the pair as if mask was linear
    #So the next line sets all the values of the mask array where the indexes are in pairs_a to True
    np.put(mask, np.ravel_multi_index(pairs_a.T, mask.shape), True)
    #This line set all the values of the mask array where the indexes are in pairs_b to False. This step is needed so that pairs in a that are also
    #in b are set to False
    np.put(mask, np.ravel_multi_index(pairs_b.T, mask.shape), False)
    if remove_flip:
        #This line does the same thing except we first flip the coordinates in pairs_b, so we get both (x, y) and (y, x) to False
        np.put(mask, np.ravel_multi_index(np.flip(pairs_b, 1).T, mask.shape), False)

    #Finally, we return the non False elements, e.g the pairs in a that are not in b.
    return np.column_stack(np.nonzero(mask))


def find_continuous_pairs(chain_id_arr, res_id_arr, atom_name_arr):
    pairs = []

    # res_id in different chains are duplicated, so loop on chains
    u_chain_id = np.unique(chain_id_arr)

    for c_id in u_chain_id:
        tmp_mask = chain_id_arr == c_id
        tmp_indices_in_pdb = np.nonzero(tmp_mask)[0]

        tmp_res_id_arr = res_id_arr[tmp_mask]
        tmp_atom_name_arr = atom_name_arr[tmp_mask]

        # check is aa or nt
        tmp_atom_name_set = set(tmp_atom_name_arr)

        if len(tmp_atom_name_set.intersection(AA_ATOMS)) > len(tmp_atom_name_set.intersection(NT_ATOMS)):
            in_res_atom_names = AA_ATOMS
        elif len(tmp_atom_name_set.intersection(AA_ATOMS)) < len(tmp_atom_name_set.intersection(NT_ATOMS)):
            in_res_atom_names = NT_ATOMS
        else:
            raise NotImplemented("Cannot determine chain is amino acid or nucleotide.")

        # find pairs
        if len(in_res_atom_names) == 1:
            u_res_id, indices_in_chain = np.unique(tmp_res_id_arr, return_index=True)
            if len(u_res_id) != np.sum(tmp_mask):
                raise ValueError(f"Found duplicate residue id in single chain {c_id}.")

            indices_in_chain_pair = np.column_stack((indices_in_chain[:-1], indices_in_chain[1:]))

            # must be adjacent on residue id
            valid_mask = np.abs(np.diff(u_res_id[indices_in_chain_pair], axis=1)) == 1

            indices_in_chain_pair = indices_in_chain_pair[valid_mask.flatten()]

            indices_in_pdb_pair = tmp_indices_in_pdb[indices_in_chain_pair]
        elif len(in_res_atom_names) > 1:

            def _cmp(a, b):
                # res_id compare
                if a[0] != b[0]:
                    return a[0] - b[0]
                else:
                    # atom_name in the same order of AA_ATOMS or NT_ATOMS
                    return in_res_atom_names.index(a[1]) - in_res_atom_names.index(b[1])

            cache = list(zip(tmp_res_id_arr, tmp_atom_name_arr, tmp_indices_in_pdb))
            sorted_cache = list(sorted(cache, key=cmp_to_key(_cmp)))

            sorted_indices_in_pdb = [item[2] for item in sorted_cache]
            sorted_res_id = [item[0] for item in sorted_cache]

            indices_in_pdb_pair = np.column_stack((sorted_indices_in_pdb[:-1], sorted_indices_in_pdb[1:]))

            valid_mask = np.abs(np.diff(np.column_stack((sorted_res_id[:-1], sorted_res_id[1:])), axis=1)) <= 1

            indices_in_pdb_pair = indices_in_pdb_pair[valid_mask.flatten()]
        else:
            raise NotImplemented("No enough atoms to construct continuous pairs.")

        pairs.append(indices_in_pdb_pair)

    pairs = np.vstack(pairs)
    return pairs



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
    print("ERR", err)
    return err

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
    return torch.mean(all_average_clahing)


def compute_loss(predicted_images, images, mask_image, latent_mean, latent_std, vae, loss_weights,
                 experiment_settings, tracking_dict, pairs_continuous_loss = None, pairs_clashing_loss = None, dists_pairs = None, predicted_structures = None, true_structure=None, device=None):
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

    ## !!!!!!!!! REPLACING THE RMSD WITH A CORRELATION COMPUTATION BE CAREFUL !!!!!!!!!!!!!!
    loss_type = experiment_settings.get("loss_type")
    if loss_type == "msd":
        print("LOSS TYPE: msd")
        rmsd = compute_rmsd(predicted_images, images)
    else:
        rmsd = calc_cor_loss(predicted_images, images, mask_image)


    KL_prior_latent = compute_KL_prior_latent(latent_mean, latent_std, experiment_settings["epsilon_kl"])
    KL_prior_mask_means = compute_KL_prior_mask(
        vae.mask_parameters, experiment_settings["mask_prior"],
        "means", epsilon_kl=experiment_settings["epsilon_kl"])

    #continuity_loss = compute_continuity_loss(predicted_structures, true_structure, device)
    continuity_loss = calc_pair_dist_loss(predicted_structures, pairs_continuous_loss, dists_pairs, type="vanilla", chain_id=None)
    if pairs_clashing_loss is None:
        clashing_loss = compute_clashing_distances(predicted_structures, device)
    else:
        clashing_loss =  calc_clash_loss(predicted_structures, pairs_clashing_loss, clash_cutoff=4.0)

    KL_prior_mask_stds = compute_KL_prior_mask(vae.mask_parameters, experiment_settings["mask_prior"],
                                               "stds", epsilon_kl=experiment_settings["epsilon_kl"])
    KL_prior_mask_proportions = compute_KL_prior_mask(vae.mask_parameters, experiment_settings["mask_prior"],
                                               "proportions", epsilon_kl=experiment_settings["epsilon_kl"])
    l2_pen = compute_l2_pen(vae)



    tracking_dict["rmsd"].append(rmsd.detach().cpu().numpy())
    tracking_dict["kl_prior_latent"].append(KL_prior_latent.detach().cpu().numpy())
    tracking_dict["kl_prior_mask_mean"].append(KL_prior_mask_means.detach().cpu().numpy())
    tracking_dict["kl_prior_mask_std"].append(KL_prior_mask_stds.detach().cpu().numpy())
    tracking_dict["kl_prior_mask_proportions"].append(KL_prior_mask_proportions.detach().cpu().numpy())
    tracking_dict["l2_pen"].append(l2_pen.detach().cpu().numpy())
    tracking_dict["continuity_loss"].append(continuity_loss.detach().cpu().numpy())
    tracking_dict["clashing_loss"].append(clashing_loss.detach().cpu().numpy())
    tracking_dict["clashing_loss"].append(clashing_loss.detach().cpu().numpy())

    loss = rmsd + loss_weights["KL_prior_latent"]*KL_prior_latent \
           + loss_weights["KL_prior_mask_mean"]*KL_prior_mask_means \
           + loss_weights["KL_prior_mask_std"] * KL_prior_mask_stds \
           + loss_weights["KL_prior_mask_proportions"] * KL_prior_mask_proportions \
           + loss_weights["l2_pen"] * l2_pen \
           + loss_weights["continuity_loss"]*continuity_loss \
           + loss_weights["clashing_loss"]*clashing_loss

    return loss
