import torch
import random
import numpy as np
from typing import Any, Tuple
import argparse
max_norm = 85
eps = 1e-8

def to_device(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor

def str2bool(v: str) -> bool:
    v = v.lower()
    if v == "true":
        return True
    elif v == "false":
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected, got" + str(v) + ".")

class LeakyClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, min: float, max: float) -> torch.Tensor:
        ctx.save_for_backward(x.ge(min) * x.le(max))
        return torch.clamp(x, min=min, max=max)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        mask, = ctx.saved_tensors
        mask = mask.type_as(grad_output)
        return grad_output * mask + grad_output * (1 - mask) * eps, None, None

def clamp(x: torch.Tensor, min: float = float("-inf"), max: float = float("+inf")) -> torch.Tensor:
    return LeakyClamp.apply(x, min, max)

def cosh(x: torch.Tensor) -> torch.Tensor:
    x = clamp(x, min=-max_norm, max=max_norm)
    return torch.cosh(x)

def save(filename, log_file, model, args, opts, epoch, entity_idxs, relation_idxs, timestamp_idxs, main_dirName):
    """Save current state to specified file"""
    log_file.write("Saving checkpoint to {}... \n".format(filename))
    model = [component.state_dict() for component in model]
    torch.save(
        {
            "type": "train",
            "epoch": epoch,
            "model": model,
            "optimizer_state_dict": [optimizer.state_dict() for optimizer in opts],
            "entity_idxs": entity_idxs,
            "relation_idxs" : relation_idxs,
            "timestamp_idxs" : timestamp_idxs,
            "learning_rate" : args.lr,
            "dim" : args.dim,
            "nneg" : args.nneg,
            "num_iterations" : args.num_iterations,
            "batch_size" : args.batch_size,
            "batch_size_eva" : args.batch_size_eva,
            "lr_cur" : args.lr_cur,
            "curvatures_fixed" : args.curvatures_fixed,
            "curvatures_trainable" : args.curvatures_trainable,
            "tr_cur" : args.trainable_curvature,
            "main_dirName" : main_dirName,
            "dataset" : args.dataset,
            "time_rescale" : args.time_rescale
        },
        filename,
    )

# # ##################Functions for spherical space ###########################################################
def sphere_sum_c(x, y, c):
    if c < 0 :
        raise ValueError("error in sphere_sum_c")
    else:
        sqxnorm_c = c * torch.sum(x * x, dim=-1, keepdim=True)
        sqynorm_c = c * torch.sum(y * y, dim=-1, keepdim=True)

    dotxy = torch.sum(x*y, dim=-1, keepdim=True)
    numerator = (1 - 2*c*dotxy - sqynorm_c)*x + (1 + sqxnorm_c)*y
    denominator = 1 - 2*c*dotxy + sqxnorm_c*sqynorm_c
    return numerator/denominator

def sphere_exp_map_c(v, c):
    normv_c = torch.clamp(torch.sqrt(torch.abs(c))*torch.norm(v, 2, dim=-1, keepdim=True), min=1e-10)
    return torch.tan(normv_c)*v/(normv_c)


def sphere_sqdist(p1, p2, c): #c is positive in projected hypersphere space, negative in poincare ball.
    sqrt_c = torch.sqrt(torch.abs(c))
    dist = torch.atan(sqrt_c * torch.norm(sphere_sum_c(-p1, p2, c), 2, dim=-1))
    sqdist = ((dist * 2 / sqrt_c) ** 2).clamp(max=75)
    return sqdist

def full_sphere_exp_map_c(x, v, c): #tangent space of an reference point x
    normv_c = torch.clamp(torch.sqrt(torch.abs(c)) * torch.norm(v, 2, dim=-1, keepdim=True), min=1e-10)
    if c < 0:
        raise ValueError("error in full_sphere_exp_map_c")
    else:
        sqxnorm_c = c * torch.sum(x * x, dim=-1, keepdim=True)
    y = torch.tan(normv_c/(1 + sqxnorm_c)) * v/(normv_c)
    return sphere_sum_c(x, y, c)

def sphere_log_map_c(v, c):
    normv = torch.clamp(torch.norm(v, 2, dim=-1, keepdim=True), 1e-10)  # we need clamp here because we need to divide the norm.
    sqrt_c = torch.sqrt(torch.abs(c))
    normv_c = sqrt_c * normv

    return 1. / sqrt_c * torch.atan(normv_c) * (v / normv)


# ############Functions for Poincare space#################################################
def artanh(x):
    return 0.5*torch.log((1+x)/(1-x))

def poincare_exp_map_c(v, c):
    normv = torch.clamp(torch.norm(v, 2, dim=-1, keepdim=True), min=1e-10) #we need clamp here because we need to divide the norm.
    normv_c = torch.clamp(torch.sqrt(c)*normv, min=1e-10)
    return torch.tanh(normv_c)*v/(normv_c)

def poincare_log_map_c(v, c):
    normv = torch.clamp(torch.norm(v, 2, dim=-1, keepdim=True), 1e-10,
                        1 - 1e-5)  # we need clamp here because we need to divide the norm.
    sqrt_c = torch.sqrt(c)
    if sqrt_c.detach().item() < 1e-10:
        raise ValueError("sqrt of curvature smaller than 1e-10")
    normv_c = torch.clamp(sqrt_c * normv, max=1 - 1e-5)

    return 1 / sqrt_c * artanh(normv_c) * v / normv

def poincare_sum_c(x, y, c):
    sqxnorm_c = torch.clamp(c*torch.sum(x * x, dim=-1, keepdim=True), 0, 1-1e-5)
    sqynorm_c = torch.clamp(c*torch.sum(y * y, dim=-1, keepdim=True), 0, 1-1e-5)
    dotxy = torch.sum(x*y, dim=-1, keepdim=True)
    numerator = (1+2*c*dotxy+sqynorm_c)*x + (1-sqxnorm_c)*y
    denominator = 1 + 2*c*dotxy + sqxnorm_c*sqynorm_c
    return numerator/denominator

def full_poincare_exp_map_c(x, v, c): #tangent space of an reference point x
    normv_c = torch.clamp(torch.sqrt(c) * torch.norm(v, 2, dim=-1, keepdim=True), min=1e-10)
    sqxnorm_c = torch.clamp(c*torch.sum(x * x, dim=-1, keepdim=True), 0, 1-1e-5) #we need clamp here because we need to divide the norm.
    y = torch.tanh(normv_c/(1-sqxnorm_c)) * v/(normv_c)
    return poincare_sum_c(x, y, c)

def poincare_sqdist(p1, p2, c): #c is positive in projected hypersphere space, negative in poincare ball.
    sqrt_c = torch.sqrt(torch.abs(c))
    dist = artanh(torch.clamp(sqrt_c * torch.norm(poincare_sum_c(-p1, p2, c), 2, dim=-1), 1e-10, 1 - 1e-5))
    sqdist = (2. * dist / sqrt_c) ** 2
    return sqdist

def poincare_cosh_sqdist(p1, p2, c): #c is positive in projected hypersphere space, negative in poincare ball.
    sqrt_c = torch.sqrt(torch.abs(c))
    dist = artanh(torch.clamp(sqrt_c * torch.norm(poincare_sum_c(-p1, p2, c), 2, dim=-1), 1e-10, 1 - 1e-5))
    cosh_sqdist = cosh(2. * dist / sqrt_c) ** 2
    return cosh_sqdist

# #################Negative Sampling Methods#################################################
# generate negative samples by corrupting head or tail with equal probabilities with checking whether false negative samples exist.
def getBatch_filter_all_dyn(quadruples, entityTotal, srt_vocab, ort_vocab = None, corrupt_head = False, mult_num = 1):
    '''
    quadruples: training batch
    entityTotal: the number of entities in the whole dataset
    corrupt_head: whether to corrupt the subject entity
    mult_num: number of negative samples
    '''
    newQuadrupleList = [corrupt_head_filter_dyn(quadruples[i], entityTotal, ort_vocab) if corrupt_head
        else corrupt_tail_filter_dyn(quadruples[i], entityTotal, srt_vocab) for i in range(len(quadruples))]
    batch_list = []
    batch_list.append(np.array(newQuadrupleList))
    if mult_num > 1:
        for i in range(0, mult_num-1):
            newQuadrupleList2 = [corrupt_head_filter_dyn(quadruples[i], entityTotal, ort_vocab) if corrupt_head
                     else corrupt_tail_filter_dyn(quadruples[i], entityTotal, srt_vocab) for i in range(len(quadruples))]
            batch_list.append(np.array(newQuadrupleList2))
        batch_list = np.stack(batch_list, axis=1) #shape: batch_size * self.nneg * 4
    return batch_list

def getBatch_filter_all_static(quadruples, entityTotal, sr_vocab, or_vocab = None, corrupt_head = False, mult_num = 1):
    '''
    quadruples: training batch
    entityTotal: the number of entities in the whole dataset
    corrupt_head: whether to corrupt the subject entity
    mult_num: number of negative samples
    '''
    newQuadrupleList = [corrupt_head_filter_static(quadruples[i], entityTotal, or_vocab) if corrupt_head
        else corrupt_tail_filter_static(quadruples[i], entityTotal, sr_vocab) for i in range(len(quadruples))]
    batch_list = []
    batch_list.append(np.array(newQuadrupleList))
    if mult_num > 1:
        for i in range(0, mult_num-1):
            newQuadrupleList2 = [corrupt_head_filter_static(quadruples[i], entityTotal, or_vocab) if corrupt_head
                     else corrupt_tail_filter_static(quadruples[i], entityTotal, sr_vocab) for i in range(len(quadruples))]
            batch_list.append(np.array(newQuadrupleList2))
        batch_list = np.stack(batch_list, axis=1) #shape: batch_size * self.nneg * 4
    return batch_list

# Change the tail of a triple randomly,
# with checking whether it is a false negative sample.
# If it is, regenerate.
def corrupt_tail_filter_dyn(quadruple, entityTotal, quadrupleDict):
    while True:
        newTail = random.randrange(entityTotal)
        if newTail not in set(quadrupleDict[(quadruple[0], quadruple[1]), quadruple[3]]):
            break
    return [quadruple[0], quadruple[1], newTail, quadruple[3]]

# Change the head of a triple randomly,
# with checking whether it is a false negative sample.
# If it is, regenerate.
def corrupt_head_filter_dyn(quadruple, entityTotal, quadrupleDict):
    while True:
        newHead = random.randrange(entityTotal)
        if newHead not in set(quadrupleDict[(quadruple[2], quadruple[1], quadruple[3])]):
            break
    return [newHead, quadruple[1], quadruple[2] , quadruple[3]]

# If it is, regenerate.
def corrupt_tail_filter_static(quadruple, entityTotal, tripleDict):
    while True:
        newTail = random.randrange(entityTotal)
        if newTail not in set(tripleDict[(quadruple[0], quadruple[1])]):
            break
    return [quadruple[0], quadruple[1], newTail, quadruple[3]]

# Change the head of a triple randomly,
# with checking whether it is a false negative sample.
# If it is, regenerate.
def corrupt_head_filter_static(quadruple, entityTotal, tripleDict):
    while True:
        newHead = random.randrange(entityTotal)
        if newHead not in set(tripleDict[(quadruple[2], quadruple[1])]):
            break
    return [newHead, quadruple[1], quadruple[2] , quadruple[3]]
