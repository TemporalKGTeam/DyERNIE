import torch.nn as nn
from utils import *
import torch.nn.functional as F

class DyERNIE_P(nn.Module):
    def __init__(self, d, dim, learning_rate, fixed_c = None, use_cosh = False, dropout = 0, vmax = 1):
        super(DyERNIE_P, self).__init__()
        self.name = 'Poincare'
        self.learning_rate = learning_rate
        self.dim = dim
        self.use_cosh = use_cosh
        self.dropout = dropout
        self.v_max = vmax
        self.curvature = to_device(torch.tensor(fixed_c, dtype=torch.double, requires_grad=False))


        self.P = nn.Parameter(to_device(torch.tensor(np.random.uniform(-1, 1, (len(d.relations), dim)), dtype=torch.double,requires_grad=True)))  # In tangent space
        self.bs = nn.Parameter(to_device(torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True)))
        self.bo = nn.Parameter(to_device(torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True)))
        self.time_emb_v = nn.Parameter(to_device(1e-3 * torch.randn((len(d.entities), dim), dtype=torch.double, requires_grad=True)))  # defined in tangent space
        self.initial_E = nn.Embedding(len(d.entities), dim, padding_idx=0)
        self.p = nn.Embedding(len(d.relations), dim, padding_idx=0)

        self.p.weight.data =  to_device(1e-3 * torch.randn((len(d.relations), dim), dtype=torch.double))
        self.initial_E.weight.data = to_device(1e-3 * torch.randn((len(d.entities), dim), dtype=torch.double))


    def emb_evolving_vanilla(self, e_idx, times, use_dropout = False):
        curvature = self.curvature
        init_embd_p = self.initial_E.weight[e_idx] #defined in hyperblic space
        init_embd_p = torch.where(torch.norm(init_embd_p, 2, dim=-1, keepdim=True) >= (1 / torch.sqrt(torch.abs(curvature.detach())).item()),
            init_embd_p / (torch.norm(init_embd_p, 2, dim=-1, keepdim=True) * torch.sqrt(torch.abs(curvature)) - 1e-5),init_embd_p)

        # ##################project the initial embeddings into the tangent space
        init_embd_e = poincare_log_map_c(init_embd_p, curvature)

        # #########all velocity vectors are defined in the tangent space, update the embeddings.
        tau = times[:, :, None]
        linear_velocities = self.time_emb_v[e_idx]

        # ###########normalize the velocity vector
        linear_velocities = torch.where(torch.norm(linear_velocities, 2, dim=-1, keepdim=True) >= (
                self.v_max / torch.sqrt(torch.abs(curvature.detach())).item()),linear_velocities * self.v_max / (torch.norm(linear_velocities, 2, dim=-1, keepdim=True) * torch.sqrt(
                                torch.abs(curvature)) - 1e-5), linear_velocities)
        emd_linear_temp =  linear_velocities * tau # batch*nneg*dim
        new_embds_e = init_embd_e + emd_linear_temp
        new_embds_e = torch.where(torch.norm(new_embds_e, 2, dim=-1, keepdim=True) >= (1 / torch.sqrt(torch.abs(curvature.detach())).item()),
                                  new_embds_e / (torch.norm(new_embds_e, 2, dim=-1, keepdim=True) - 1e-5) / torch.sqrt(torch.abs(curvature)), new_embds_e)

        # ##################drift in the tangent space
        new_embds_p = poincare_exp_map_c(new_embds_e, curvature)
        new_embds_p = torch.where(torch.norm(new_embds_p, 2, dim=-1, keepdim=True) >= (1/torch.sqrt(torch.abs(curvature.detach())).item()),
                                  new_embds_p/(torch.norm(new_embds_p, 2, dim=-1, keepdim=True) * torch.sqrt(torch.abs(curvature)) - 1e-5), new_embds_p)

        if use_dropout:
            new_embds_p = F.dropout(new_embds_p, p=self.dropout, training=self.training)

        return new_embds_p

    def forward(self, u_idx, r_idx, v_idx, t):
        curvature = self.curvature
        P = self.P[r_idx]
        u = self.emb_evolving_vanilla(u_idx, t)
        v = self.emb_evolving_vanilla(v_idx, t)
        p = self.p.weight[r_idx]
        p = torch.where(torch.norm(p, 2, dim=-1, keepdim=True) >= (1 / torch.sqrt(torch.abs(curvature.detach())).item()),
                p / (torch.norm(p, 2, dim=-1, keepdim=True) * torch.sqrt(torch.abs(curvature)) - 1e-5), p)

        # Moebius matrix-vector multiplication
        # map the original subject entity embedding to the tangent space of the Poincaré ball at 0
        u_e = poincare_log_map_c(u, curvature)

        #transforming it by the diagonal relation matrix
        u_P = u_e * P

        # project back to the poincare ball
        u_m = poincare_exp_map_c(u_P, curvature)

        # Moebius addition
        v_m = poincare_sum_c(v, p, curvature)

        # Poincare ball constraints
        u_m = torch.where(
            torch.norm(u_m, 2, dim=-1, keepdim=True) >= (1 / torch.sqrt(torch.abs(curvature.detach())).item()),
            u_m / (torch.norm(u_m, 2, dim=-1, keepdim=True) * torch.sqrt(torch.abs(curvature)) - 1e-5), u_m)
        v_m = torch.where(
            torch.norm(v_m, 2, dim=-1, keepdim=True) >= (1 / torch.sqrt(torch.abs(curvature.detach())).item()),
            v_m / (torch.norm(v_m, 2, dim=-1, keepdim=True) * torch.sqrt(torch.abs(curvature)) - 1e-5), v_m)

        # compute the distance between two points on the Poincare ball along a geodesic.
        if self.use_cosh:
            sqdist = poincare_cosh_sqdist(u_m, v_m, curvature)
        else:
            sqdist = poincare_sqdist(u_m, v_m, curvature)
        predictions = -sqdist + self.bs[u_idx] + self.bo[v_idx]
        return predictions



class DyERNIE_E(torch.nn.Module):
    def __init__(self, d, dim, learning_rate, use_cosh = False, dropout = 0):
        super(DyERNIE_E, self).__init__()
        self.name = 'Euclidean'
        self.learning_rate = learning_rate
        self.dim = dim
        self.use_cosh = use_cosh
        self.dropout = dropout
        self.curvature = to_device(torch.tensor(0., dtype=torch.double, requires_grad=False))

        r = 6 / np.sqrt(dim)
        self.P = nn.Parameter(to_device(torch.tensor(np.random.uniform(-r, r, (len(d.relations), dim)), dtype=torch.double, requires_grad=True)))
        self.bs = nn.Parameter(to_device(torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True)))
        self.bo = nn.Parameter(to_device(torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True)))

        self.initial_E_euc = nn.Embedding(len(d.entities), dim, padding_idx=0)
        self.initial_E_euc.weight.data = to_device(torch.tensor(np.random.uniform(-r, r, (len(d.entities), dim)), dtype=torch.double))
        self.time_emb_v =nn.Parameter(to_device(torch.tensor(np.random.uniform(-r, r, (len(d.entities), dim)), dtype=torch.double, requires_grad=True)))

        ####relation vector
        self.p_euc = nn.Embedding(len(d.relations), dim, padding_idx=0)
        self.p_euc.weight.data = to_device(1e-3 * torch.randn((len(d.relations), dim), dtype=torch.double))

    def emb_evolving_vanilla(self, e_idx, times, use_dropout = False):
        init_embd_e = self.initial_E_euc.weight[e_idx]
        linear_velocities = self.time_emb_v[e_idx]

        # #########all velocity vectors are defined in the tangent space, update the embeddings
        emd_linear_temp = linear_velocities * times[:, :, None] # batch*nneg*dim

        # ##################drift in the tangent space
        new_embds_e = init_embd_e + emd_linear_temp

        if use_dropout:
            new_embds_e = F.dropout(new_embds_e, p=self.dropout, training=self.training)

        return new_embds_e

    def forward(self, u_idx, r_idx, v_idx, t):
        P = self.P[r_idx]
        u_e = self.emb_evolving_vanilla(u_idx, t)
        v = self.emb_evolving_vanilla(v_idx, t)
        p = self.p_euc.weight[r_idx]

        # transforming it by the diagonal relation matrix
        u_m = u_e * P

        # addition
        v_m = v + p

        # compute the distance between two points.
        sqdist =  (u_m - v_m).pow(2).sum(dim=-1)
        predictions = -sqdist + self.bs[u_idx] + self.bo[v_idx]
        return predictions

class DyERNIE_S(torch.nn.Module):
    def __init__(self, d, dim, learning_rate, fixed_c = None):
        super(DyERNIE_S, self).__init__()
        self.name = 'Hypersphere'
        self.learning_rate = learning_rate
        self.p = nn.Embedding(len(d.relations), dim, padding_idx=0)
        self.P = nn.Parameter(to_device(
            torch.tensor(np.random.uniform(-1, 1, (len(d.relations), dim)), dtype=torch.double,
                         requires_grad=True)))  # in the tangent space
        self.bs = nn.Parameter(to_device(torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True)))
        self.bo = nn.Parameter(to_device(torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True)))
        self.curvature = to_device(torch.tensor(fixed_c, dtype=torch.double, requires_grad=False))
        self.initial_E = nn.Embedding(len(d.entities), dim,
                                       padding_idx=0)  # the initial entity embeddings are learned during training.
        self.p.weight.data = to_device(1e-3 * torch.randn((len(d.relations), dim), dtype=torch.double))
        self.initial_E.weight.data = to_device(1e-3 * torch.randn((len(d.entities), dim), dtype=torch.double))
        self.time_emb_v = nn.Parameter(to_device(1e-3 * torch.randn((len(d.entities), dim), dtype=torch.double,
                                                                    requires_grad=True)))  # defined in the Euclidian space

    def emb_evolving_vanilla(self, e_idx, times):
        init_embd_p = self.initial_E.weight[e_idx]  # defined in the spherical space
        linear_velocities = self.time_emb_v[e_idx]
        curvature = self.curvature

        # #########all velocity vectors are defined in the tangent space, update embeddings
        tau = times[:, :, None]
        emd_linear_temp = linear_velocities * tau  # shape: batch*nneg*dim

        # ##################project the initial embedding into the tangent space
        init_embd_e = sphere_log_map_c(init_embd_p, curvature)

        # ##################drift in the tangent space
        new_embds_e = init_embd_e + emd_linear_temp  # + emd_season_temp
        new_embds_p = sphere_exp_map_c(new_embds_e, curvature)
        return new_embds_p

    def forward(self, u_idx, r_idx, v_idx, t):
        curvature = self.curvature

        # Dynamic Embeddings
        u = self.emb_evolving_vanilla(u_idx, t)
        v = self.emb_evolving_vanilla(v_idx, t)
        P = self.P[r_idx]
        p = self.p.weight[r_idx]

        # Moebius matrix-vector multiplication
        # map the original subject entity embedding to the tangent space of the Poincaré ball at 0
        u_e = sphere_log_map_c(u, curvature)

        #transforming it by the diagonal relation matrix
        u_P = u_e * P

        # project back to the poincare ball
        u_m = sphere_exp_map_c(u_P, curvature)

        # Moebius addition
        v_m = sphere_sum_c(v, p, curvature)

        # compute the distance between two points on the Poincare ball along a geodesic.
        sqdist = sphere_sqdist(u_m, v_m, curvature)
        predictions = -sqdist + self.bs[u_idx] + self.bo[v_idx]

        return predictions
