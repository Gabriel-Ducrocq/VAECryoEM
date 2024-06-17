import torch
import numpy as np


class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder, device, mask_start_values, N_domains=6, N_residues=1006, tau_mask=0.05,
                 latent_dim = None, latent_type="continuous"):
        super(VAE, self).__init__()
        assert latent_type in ["continuous", "categorical"]
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.N_domains = N_domains
        self.N_residues = N_residues
        self.tau_mask = tau_mask
        self.latent_type = latent_type
        self.latent_dim = latent_dim

        self.residues = torch.arange(0, self.N_residues, 1, dtype=torch.float32, device=device)[:, None]

        assert mask_start_values["type"] == "uniform", "Currently, only uniform initialization of the segmentation available."
        bound_0 = self.N_residues/N_domains
        self.segments_means = torch.nn.Parameter(data=torch.tensor(np.array([bound_0/2 + i*bound_0 for i in range(N_domains)]), dtype=torch.float32, device=device)[None, :],
                                                  requires_grad=True)

        self.segments_std = torch.nn.Parameter(data= torch.tensor(np.ones(N_domains)*bound_0, dtype=torch.float32, device=device)[None,:],
                                                requires_grad=True)


        self.segments_proportions = torch.nn.Parameter(
            data=torch.tensor(np.ones(N_domains) * 0, dtype=torch.float32, device=device)[None, :],
            requires_grad=True)


        self.segments_parameters = {"means":self.segments_means, "stds":self.segments_std, "proportions":self.segments_proportions}
        self.elu = torch.nn.ELU()

    def sample_mask(self, N_batch):
        """
        Samples a mask
        :return: torch.tensor(N_batch, N_residues, N_domains) values of the mask
        """
        cluster_proportions = self.segments_proportions.repeat(N_batch, 1)
        cluster_means = self.segments_means.repeat(N_batch, 1)
        cluster_std = self.segments_std.repeat(N_batch, 1)
        proportions = torch.softmax(cluster_proportions, dim=-1)

        log_num = -0.5*(self.residues[None, :, :] - cluster_means[:, None, :])**2/cluster_std[:, None, :]**2 + \
              torch.log(proportions[:, None, :])

        segmentation = torch.softmax(log_num / self.tau_mask, dim=-1)

        return segmentation

    def sample_latent(self, images):
        """
        Samples latent variables given an image
        :param images: torch.tensor(N_batch, N_pix_x, N_pix_y)
        :return: torch.tensor(N_batch, latent_dim) latent variables,
                torch.tensor(N_batch, latent_dim) latent_mean,
                torch.tensor(N_batch, latent_dim) latent std if latent_type is "continuous"
                else
                torch.tensor(N_batch, 1) sampled latent variable per batch
                torch.tensor(N_batch, latent_dim) log probabilities of the multinomial.
        """
        if self.latent_type == "continuous":
            latent_mean, latent_std = self.encoder(images)
            latent_variables = latent_mean + torch.randn_like(latent_mean, dtype=torch.float32, device=self.device)\
                                *latent_std

            return latent_variables, latent_mean, latent_std
        else:
            log_distribution_latent = self.encoder(images)
            distribution_latent = torch.softmax(log_distribution_latent, dim=-1)
            latent_variable = torch.multinomial(distribution_latent, 1)
            return latent_variable, log_distribution_latent, None

    def decode(self, latent_variables):
        """
        Decode the latent variables
        :param latent_variables: torch.tensor(N_batch, latent_dim)
        :return: torch.tensor(N_batch, N_domains, 4) quaternions, torch.tensor(N_batch, N_domains, 3) translations
                OR torch.tensor(N_latent_dim, N_domains, 4)
        """
        N_batch = latent_variables.shape[0]
        if self.latent_type == "continuous":
            transformations = self.decoder(latent_variables)
            transformations_per_domain = torch.reshape(transformations, (N_batch, self.N_domains, 6))
            ones = torch.ones(size=(N_batch, self.N_domains, 1), device=self.device)
            quaternions_per_domain = torch.concat([ones, transformations_per_domain[:, :, 3:]], dim=-1)
            translations_per_domain = transformations_per_domain[:, :, :3]
        else:
            latent_variables = torch.tensor([latent_variable for latent_variable in range(self.latent_dim)],
                                           dtype=torch.float32, device=self.device)[:, None]
            transformations = self.decoder(latent_variables)
            transformations_per_domain = torch.reshape(transformations, (self.latent_dim, self.N_domains, 6))
            ones = torch.ones(size=(self.latent_dim, self.N_domains, 1), device=self.device)
            quaternions_per_domain = torch.concat([ones, transformations_per_domain[:, :, 3:]], dim=-1)
            translations_per_domain = transformations_per_domain[:, :, :3]

        return quaternions_per_domain, translations_per_domain



