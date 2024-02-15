import torch
import numpy as np
from operator import itemgetter
from pytorch3d.transforms import matrix_to_rotation_6d


class VAE(torch.nn.Module):
    def __init__(self, device, mask_start_values, N_images=150000, N_domains=6, N_residues=1006, tau_mask=0.05,
                 latent_dim = None, latent_type="continuous", representation="r6"):
        super(VAE, self).__init__()
        assert latent_type in ["continuous", "categorical"]
        self.device = device
        self.N_domains = N_domains
        self.N_residues = N_residues
        self.tau_mask = tau_mask
        self.latent_type = latent_type
        self.latent_dim = latent_dim
        self.representation = representation

        self.residues = torch.arange(0, self.N_residues, 1, dtype=torch.float32, device=device)[:, None]

        if mask_start_values["type"] == "uniform":
            bound_0 = self.N_residues/N_domains
            self.mask_means_mean = torch.nn.Parameter(data=torch.tensor(np.array([bound_0/2 + i*bound_0 for i in range(N_domains)]), dtype=torch.float32, device=device)[None, :],
                                                      requires_grad=True)
            self.mask_means_std = torch.nn.Parameter(data= torch.tensor(np.ones(N_domains)*10.0, dtype=torch.float32, device=device)[None,:],
                                                    requires_grad=True)
            self.mask_std_mean = torch.nn.Parameter(data= torch.tensor(np.ones(N_domains)*bound_0, dtype=torch.float32, device=device)[None,:],
                                                    requires_grad=True)

            self.mask_std_std = torch.nn.Parameter(
                data=torch.tensor(np.ones(N_domains) * 10.0, dtype=torch.float32, device=device)[None, :],
                requires_grad=True)

            self.mask_proportions_mean = torch.nn.Parameter(
                data=torch.tensor(np.ones(N_domains) * 0, dtype=torch.float32, device=device)[None, :],
                requires_grad=True)

            self.mask_proportions_std = torch.nn.Parameter(
                data=torch.tensor(np.ones(N_domains), dtype=torch.float32, device=device)[None, :],
                requires_grad=True)



        else:
            self.mask_means_mean = torch.nn.Parameter(data=torch.tensor(mask_start_values["clusters_mean"]["mean"], dtype=torch.float32,device=device)[None, :],
                                                    requires_grad=True)

            self.mask_means_std = torch.nn.Parameter(data=torch.tensor(mask_start_values["clusters_mean"]["std"], dtype=torch.float32, device=device)[None, :],
                                                  requires_grad=True)

            self.mask_std_mean = torch.nn.Parameter(data=torch.tensor(mask_start_values["clusters_std"]["mean"], dtype=torch.float32, device=device)[None, :],
                                                  requires_grad=True)

            self.mask_std_std = torch.nn.Parameter(data=torch.tensor(mask_start_values["clusters_std"]["std"], dtype=torch.float32, device=device)[None, :],
                                                  requires_grad=True)

            self.mask_proportions_mean = torch.nn.Parameter(torch.tensor(mask_start_values["clusters_proportions"]["mean"], dtype=torch.float32, device=device)[None, :],
                                                          requires_grad=True)

            self.mask_proportions_std = torch.nn.Parameter(torch.tensor(mask_start_values["clusters_proportions"]["std"], dtype=torch.float32, device=device)[None, :],
                               requires_grad=True)

        self.mask_parameters = {"means":{"mean":self.mask_means_mean, "std":self.mask_means_std},
                                   "stds":{"mean":self.mask_std_mean, "std":self.mask_std_std},
                                   "proportions":{"mean":self.mask_proportions_mean, "std":self.mask_proportions_std}}

        self.elu = torch.nn.ELU()


        self.translation_per_domain = torch.nn.Parameter(data=torch.zeros((N_images, N_domains, 3), dtype=torch.float32, device=self.device), requires_grad=True) 
        if representation =="r6":
            self.rotation_per_domain = torch.nn.Parameter(data=torch.tensor([1., 0., 0., 0., 1., 0.], dtype=torch.float32, device=self.device).repeat(N_images, N_domains, 1), requires_grad=True)
        elif representation == "axis_angle":
            self.rotation_per_domain = torch.nn.Parameter(data=torch.tensor([0., 0.,  0.], dtype=torch.float32, device=self.device).repeat(N_images, N_domains, 1), requires_grad=True)



    def sample_mask(self, N_batch):
        """
        Samples a mask
        :return: torch.tensor(N_batch, N_residues, N_domains) values of the mask
        """
        #cluster_proportions = torch.randn(self.N_domains, device=self.device)*self.mask_proportions_std + self.mask_proportions_mean
        #cluster_means = torch.randn(self.N_domains, device=self.device)*self.mask_means_std + self.mask_means_mean
        #cluster_std = self.elu(torch.randn(self.N_domains, device=self.device)*self.mask_std_std + self.mask_std_mean) + 1
        #proportions = torch.softmax(cluster_proportions, dim=1)
        #log_num = -0.5*(self.residues - cluster_means)**2/cluster_std**2 + \
        #      torch.log(proportions)

        #mask = torch.softmax(log_num/self.tau_mask, dim=1)
        cluster_proportions = torch.randn((N_batch, self.N_domains),
                                          device=self.device) * self.mask_proportions_std+ self.mask_proportions_mean
        cluster_means = torch.randn((N_batch, self.N_domains), device=self.device) * self.mask_means_std+ self.mask_means_mean
        cluster_std = self.elu(torch.randn((N_batch, self.N_domains), device=self.device)*self.mask_std_std + self.mask_std_mean) + 1
        proportions = torch.softmax(cluster_proportions, dim=-1)
        log_num = -0.5*(self.residues[None, :, :] - cluster_means[:, None, :])**2/cluster_std[:, None, :]**2 + \
              torch.log(proportions[:, None, :])

        mask = torch.softmax(log_num / self.tau_mask, dim=-1)
        return mask


    def batch_transformations(self, indexes):
        return self.rotation_per_domain[indexes], self.translation_per_domain[indexes]




