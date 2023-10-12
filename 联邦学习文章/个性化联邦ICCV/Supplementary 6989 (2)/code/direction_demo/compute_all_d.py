from cal_d import get_FedPG_d, get_d_mgdaplus_d, get_FedFV_d
import torch
import copy
import numpy as np

torch.set_default_dtype(torch.float64)

device = "cpu"

grad_mat = torch.Tensor([[1.0, -0.3, -0.0],
                         [-1.0,  1.0, -0.2],
                         [-1.0, -1.0,  1.0]])

l_locals = torch.Tensor([1.0, 2.0, 3.0])
print(grad_mat)
print()


alpha = 0
d1, descent_flag = get_FedFV_d(
    copy.deepcopy(grad_mat), l_locals, alpha, device)
d1 /= torch.norm(d1)
c = -(grad_mat @ d1.T)
if not torch.all(c <= 1e-5):
    print('FedFV: not the descent direction')
    print(c)
print('d: ', d1)
print()


epsilon = 0.1
lambda0 = np.array([1/3, 1/3, 1/3])
d2, descent_flag = get_d_mgdaplus_d(copy.deepcopy(
    grad_mat/torch.norm(grad_mat, dim=1).reshape(-1, 1)), epsilon, lambda0, device)
d2 /= torch.norm(d2)
c = -(grad_mat @ d2.T)
if not torch.all(c <= 1e-5):
    print('FedMGDA+: not the descent direction')
    print(c)
print('d: ', d2)
print()


weights = torch.Tensor([1/3, 1/3, 1/3]).to(device)
grad_mat_normalized = grad_mat / torch.norm(grad_mat, dim=1).reshape(-1, 1)
d3 = weights @ grad_mat
d3 /= torch.norm(d3)
c = -(grad_mat @ d3.T)
if not torch.all(c <= 1e-5):
    print('FedSGD: not the descent direction')
    print(c)
print('d: ', d3)
print()


alpha = 11.25
p = torch.Tensor([1.0, 1.0, 1.0])
p /= torch.norm(p)
grad_local_norm = torch.norm(grad_mat, dim=1)
miu = torch.mean(grad_local_norm)
d, Q, g, fair_grad = get_FedPG_d(copy.deepcopy(
    grad_mat), l_locals, None, p, miu, device)
print('fair grad: ', fair_grad)
d = d / torch.norm(d)
c = -(grad_mat @ d.T)
print(c)
if not torch.all(c <= 1e-5):
    print('FedPG: not the descent direction')
    print(c)
print('d: ', d)
print()

d_all = np.vstack([d1, d2, d3, d])
print('[', end='')
for i in range(d_all.shape[0]):
    print('[', end='')
    for j in range(d_all.shape[1]):
        print(d_all[i, j], ', ', end='')
    print('],', end='')
    print()
print(']')
