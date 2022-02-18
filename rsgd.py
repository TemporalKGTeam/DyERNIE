from torch.optim.optimizer import Optimizer
from utils import *

def euclidean_update_c(p, d_p, lr):
    p.data = p.data - lr * d_p
    return p.data

def riemannian_grad_c(p, d_p, c, model_name):
    if model_name is 'Poincare':
        p_sqnorm_c = torch.clamp(c * torch.sum(p.data ** 2, dim=-1, keepdim=True), 0, 1 - 1e-5)
        d_p = d_p * ((1 - p_sqnorm_c) ** 2 / 4).expand_as(d_p)
    elif model_name is 'Hypersphere':
        p_sqnorm_c = c * torch.sum(p.data ** 2, dim=-1, keepdim=True)
        d_p = d_p * ((1 + p_sqnorm_c) ** 2 / 4).expand_as(d_p)
    else:
        raise ValueError("This model is not implemented.")
    return d_p

def riemannian_update_c(p, d_p, lr, cur, model_name):
    v = -lr * d_p
    if model_name is 'Poincare':
        p.data = full_poincare_exp_map_c(p.data, v, cur)
    elif model_name is 'Hypersphere':
        p.data = full_sphere_exp_map_c(p.data, v, cur)
    return p.data

class RiemannianSGD(Optimizer):
    def __init__(self, params, param_names=[]):
        defaults = dict()
        super(RiemannianSGD, self).__init__(params, defaults)
        self.param_names = param_names

    def step(self, cur, model_name, lr_cur, lr=None):
        loss = None
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if self.param_names[i] in ["initial_Eh.weight", "rvh.weight", "initial_Eh_static.weight", "initial_Eh_tail.weight", "initial_Eh_tail_static.weight"]:
                    d_p = riemannian_grad_c(p, d_p, cur, model_name)
                    p.data = riemannian_update_c(p, d_p, lr, cur, model_name)
                elif self.param_names[i] in ["curvature_latent"]:
                    p.data = euclidean_update_c(p, d_p, lr_cur)
                elif self.param_names[i] in ["curvature"]:
                    raise ValueError('Riemannian object has no attribute curvature')
                else:
                    p.data = euclidean_update_c(p, d_p, lr)
        return loss


