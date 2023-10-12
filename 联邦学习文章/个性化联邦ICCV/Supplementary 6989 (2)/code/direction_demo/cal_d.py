import torch
import numpy as np
import cvxopt
from cvxopt import matrix
import os
import copy
import math
import random
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    P = 0.5 * (P + P.T)
    P = P.astype(np.double)
    q = q.astype(np.double)
    args = [matrix(P), matrix(q)]
    if G is not None:
        args.extend([matrix(G), matrix(h)])
        if A is not None:
            args.extend([matrix(A), matrix(b)])
    sol = cvxopt.solvers.qp(*args)
    return np.array(sol['x']).reshape((P.shape[1],))


def setup_qp_and_solve(vec):

    P = np.dot(vec, vec.T)
    n = P.shape[0]
    q = np.zeros(n)
    G = - np.eye(n)
    h = np.zeros(n)
    A = np.ones((1, n))
    b = np.ones(1)
    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt_solve_qp(P, q, G, h, A, b)
    return sol


def get_MGDA_d(grads, device):
    vec = grads
    sol = setup_qp_and_solve(vec.cpu().detach().numpy())
    sol = torch.from_numpy(sol).to(device)

    d = torch.matmul(sol, grads)

    descent_flag = 1
    c = - (grads @ d)
    if not torch.all(c <= 1e-6):
        descent_flag = 0
    return d, descent_flag


def quadprog(P, q, G, h, A, b):
    P = cvxopt.matrix(P.tolist())
    q = cvxopt.matrix(q.tolist(), tc='d')
    G = cvxopt.matrix(G.tolist())
    h = cvxopt.matrix(h.tolist())
    A = cvxopt.matrix(A.tolist())
    b = cvxopt.matrix(b.tolist(), tc='d')
    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(P, q.T, G.T, h.T, A.T, b)
    return np.array(sol['x'])


def setup_qp_and_solve_for_mgdaplus(vec, epsilon, lambda0):

    P = np.dot(vec, vec.T)
    n = P.shape[0]
    q = np.array([[0] for i in range(n)])

    A = np.ones(n).T
    b = np.array([1])

    lb = np.array([max(0, lambda0[i] - epsilon) for i in range(n)])
    ub = np.array([min(1, lambda0[i] + epsilon) for i in range(n)])
    G = np.zeros((2 * n, n))
    for i in range(n):
        G[i][i] = -1
        G[n + i][i] = 1
    h = np.zeros((2 * n, 1))
    for i in range(n):
        h[i] = -lb[i]
        h[n + i] = ub[i]
    sol = quadprog(P, q, G, h, A, b).reshape(-1)
    return sol


def get_d_mgdaplus_d(grads, epsilon, lambda0, device):
    vec = grads
    sol = setup_qp_and_solve_for_mgdaplus(
        vec.cpu().detach().numpy(), epsilon, lambda0)

    sol = torch.from_numpy(sol).to(device)
    d = torch.matmul(sol, grads)

    descent_flag = 1
    c = -(grads @ d)
    if not torch.all(c <= 1e-5):
        descent_flag = 0

    return d, descent_flag


def get_FedFV_d(grads, value, alpha, device):
    grads = [grads[i, :] for i in range(grads.shape[0])]

    order_grads = copy.deepcopy(grads)
    order = [_ for _ in range(len(order_grads))]

    tmp = sorted(list(zip(value, order)), key=lambda x: x[0])
    order = [x[1] for x in tmp]

    keep_original = []
    if alpha > 0:
        keep_original = order[math.ceil((len(order) - 1) * (1 - alpha)):]

    g_locals_L2_norm_square_list = []
    for g_local in grads:
        g_locals_L2_norm_square_list.append(torch.norm(g_local)**2)

    for i in range(len(order_grads)):
        if i in keep_original:
            continue
        for j in order:
            if j == i:
                continue
            else:

                dot = grads[j] @ order_grads[i]
                if dot < 0:
                    order_grads[i] = order_grads[i] - dot / \
                        g_locals_L2_norm_square_list[j] * grads[j]

    weights = torch.Tensor([1 / len(order_grads)] *
                           len(order_grads)).to(device)
    gt = weights @ torch.stack([order_grads[i]
                               for i in range(len(order_grads))])

    grads = torch.stack(grads)
    c = -(grads @ gt)
    descent_flag = 1
    if not torch.all(c <= 1e-5):
        descent_flag = 0

    return gt, descent_flag


def get_FedPG_d(grads, value, add_grads, prefer_vec, miu, device):
    value_norm = torch.norm(value)

    Q = grads

    if add_grads is not None:
        add_grads = add_grads / \
            torch.norm(add_grads, dim=1).reshape(-1, 1) * miu
        Q = torch.vstack([Q, add_grads])
    g = Q

    h_vec = (value @ prefer_vec * value / value_norm -
             prefer_vec * value_norm) / (value_norm**2)
    h_vec = h_vec.reshape(1, -1)
    fair_grad = h_vec @ grads
    fair_grad = fair_grad / torch.norm(fair_grad) * miu
    Q = torch.cat((Q, fair_grad))
    if grads.shape[0] == 1 and add_grads is None:
        d = grads.reshape(-1)
        return d, Q, g, fair_grad
    sol = setup_qp_and_solve(Q.cpu().detach().numpy())
    sol = torch.from_numpy(sol).to(device)
    d = sol @ Q
    return d, Q, g, fair_grad
