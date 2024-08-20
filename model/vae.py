import torch
import numpy as np


class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder, device, mask_start_values, N_chains, N_domains, N_residues, tau_mask=0.05,
                 latent_dim = None, latent_type="continuous", chain_ids=["A"]):
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
        self.N_chains = N_chains
        self.residues = {}

        self.dict_segments_means_means = torch.nn.ParameterDict({})
        self.dict_segments_means_stds = torch.nn.ParameterDict({})
        self.dict_segments_stds_means = torch.nn.ParameterDict({})
        self.dict_segments_stds_stds = torch.nn.ParameterDict({})
        self.dict_segments_proportions_means = torch.nn.ParameterDict({})
        self.dict_segments_proportions_stds = torch.nn.ParameterDict({})
        self.chain_ids = chain_ids

        for n_chain in self.chain_ids:
            assert mask_start_values["type"] == "uniform", "Currently, only uniform initialization of the segmentation available."
            bound_0 = self.N_residues[f"chain_{n_chain}"]/N_domains[f"chain_{n_chain}"]
            segments_means = torch.nn.Parameter(data=torch.tensor(np.array([bound_0/2 + i*bound_0 for i in range(N_domains[f"chain_{n_chain}"])]), dtype=torch.float32, device=device)[None, :],
                                                      requires_grad=True)

            segments_std = torch.nn.Parameter(data= torch.tensor(np.ones(N_domains[f"chain_{n_chain}"])*bound_0, dtype=torch.float32, device=device)[None,:],
                                                    requires_grad=True)


            segments_proportions = torch.nn.Parameter(
                data=torch.tensor(np.ones(N_domains[f"chain_{n_chain}"]) * 0, dtype=torch.float32, device=device)[None, :],
                requires_grad=True)


            self.dict_segments_means_means[f"chain_{n_chain}"] = segments_means
            self.dict_segments_stds_means[f"chain_{n_chain}"] = segments_stds
            self.dict_segments_proportions_means[f"chain_{n_chain}"] = segments_proportions
            self.residues[f"chain_{n_chain}"] = torch.arange(0, self.N_residues[f"chain_{n_chain}"], 1, dtype=torch.float32, device=device)[:, None]
            #self.segments_parameters[f"chain_{n_chain}"] = {"means":segments_means, "stds":segments_std, "proportions":segments_proportions}

        self.elu = torch.nn.ELU()

    def sample_mask(self, N_batch):
        """
        Samples a mask
        :return: dictionnary of segments for each chain: torch.tensor(N_batch, N_residues_chains, N_domains_chains) values of the segmentation
        """
        chain_segments = {}
        for n_chain in self.chain_ids:
            cluster_proportions = self.dict_segments_proportions_means[f"chain_{n_chain}"]
            cluster_means = self.dict_segments_means_means[f"chain_{n_chain}"]
            cluster_std = self.dict_segments_stds_means[f"chain_{n_chain}"]
            #cluster_proportions = self.segments_parameters[f"chain_{n_chain}"]["proportions"].repeat(N_batch, 1)
            #cluster_means = self.segments_parameters[f"chain_{n_chain}"]["means"].repeat(N_batch, 1)
            #cluster_std = self.segments_parameters[f"chain_{n_chain}"]["stds"].repeat(N_batch, 1)
            proportions = torch.softmax(cluster_proportions, dim=-1)

            log_num = -0.5*(self.residues[f"chain_{n_chain}"][None, :, :] - cluster_means[:, None, :])**2/cluster_std[:, None, :]**2 + \
                  torch.log(proportions[:, None, :])

            segmentation = torch.softmax(log_num / self.tau_mask, dim=-1)
            chain_segments[f"chain_{n_chain}"] = segmentation

        return chain_segments

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
        transformations = self.decoder(latent_variables)
        transformations_per_domain = torch.reshape(transformations, (N_batch, self.total_n_domains, 6))
        ones = torch.ones(size=(N_batch, self.total_n_domains, 1), device=self.device)
        quaternions_per_domain = torch.concat([ones, transformations_per_domain[:, :, 3:]], dim=-1)
        translations_per_domain = transformations_per_domain[:, :, :3]

        quaternions_per_domain_per_chain = {}
        translations_per_domain_per_chain = {}
        n_domains = 0
        for n_chain in self.chain_ids:
            n_dom_chain = self.N_domains[f"chain_{n_chain}"]
            quaternions_per_domain_per_chain[f"chain_{n_chain}"] = quaternions_per_domain[:, n_domains:n_domains+n_dom_chain, :]
            translations_per_domain_per_chain[f"chain_{n_chain}"] = translations_per_domain[:, n_domains:n_domains+n_dom_chain, :]
            n_domains += n_dom_chain


        return quaternions_per_domain_per_chain, translations_per_domain_per_chain



