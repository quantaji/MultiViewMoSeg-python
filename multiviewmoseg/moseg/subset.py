from typing import List, Union

import numpy as np
from scipy.linalg import svd


def compute_subset_loss_and_itermediate_variables(
    U_subset: np.ndarray,
    L: np.ndarray,
    gamma: float,
):
    num_kernels = U_subset.shape[0]

    K_hat = np.array([U_subset[i] @ U_subset[i].T for i in range(num_kernels)])

    L_tilde = L.copy()

    for i in range(num_kernels):
        # assume i=0 is the initial one
        # i = num_kernels - 1 is the last one
        Q_i: np.ndarray = None

        if i == 0:
            tmp_K_neg = K_hat[i + 1]
            Q_i = tmp_K_neg * (tmp_K_neg < 0)

        elif i == num_kernels - 1:
            tmp_K_pos = K_hat[i - 1]
            Q_i = tmp_K_pos * (tmp_K_pos > 0)

        else:
            tmp_K_neg = K_hat[i + 1]
            tmp_K_pos = K_hat[i - 1]
            Q_i = tmp_K_pos * (tmp_K_pos > 0) + tmp_K_neg * (tmp_K_neg < 0)

        L_tilde[i] = L[i] - gamma * Q_i
        L_tilde[i] = (L_tilde[i] + L_tilde[i].T) / 2

    loss = sum([np.trace(U_subset[i].T @ L_tilde[i] @ U_subset[i]) for i in range(num_kernels)])

    return loss, L_tilde


def subset_eig(
    kernels: Union[np.ndarray, List[np.ndarray]],
    num_motion: int,
    gamma: float,
    epsilon: float = 1e-8,
    max_iter: int = 30,
):
    """
    Input:
        kernels (list of np.ndarray): List of kernel matrices, of shape (V, N, N), v is the number of kernels
        num_motion (int): Number of motions
        gamma (float): subset-regularization parameter
        epsilon (float): Convergence threshold
        max_iter (int): Maximum number of iterations

    Returns:
        U_CoReg (np.ndarray): Co-regularized embeddings
        itr (int): Number of iterations performed
        loss (list of float): Loss values per iteration
        exitinfo (dict): Information about exit status
    """

    num_kernels = len(kernels)
    L, U = [], []

    # Intialize Each Spectral Embedding
    for i in range(num_kernels):
        print(i)
        K_i = kernels[i]  # (N, N)
        D_i = np.sqrt(np.sum(K_i, axis=1, keepdims=True)) + 1e-8

        L_i = np.eye(D_i.shape[0]) - K_i / D_i / D_i.T
        L.append(L_i)

        print("asdfasdfa")

        # U_tmp, _, _ = np.linalg.svd(L_i)
        print(L_i.shape)
        U_tmp, _, _ = svd(L_i)
        U_i = U_tmp[:, -num_motion:]
        U.append(U_i)

    # start subset Constrained Clustering
    U_subset = np.array(U)  # (v, N, num_motion)
    L = np.array(L)  # (v, N, N)

    loss, L_tilde = compute_subset_loss_and_itermediate_variables(
        U_subset=U_subset,
        L=L,
        gamma=gamma,
    )

    losses = [loss]

    exitinfo = {"reason": "timeout"}

    for iteration in range(max_iter):

        print(iteration)

        for i in range(num_kernels):
            D_i, vec_i = np.linalg.eigh(L_tilde[i])
            idx = np.argsort(D_i)

            U_subset[i] = vec_i[:, idx[:num_motion]]

        loss, L_tilde = compute_subset_loss_and_itermediate_variables(
            U_subset=U_subset,
            L=L,
            gamma=gamma,
        )

        losses.append(loss)

        if abs(losses[-2] - losses[-1]) < epsilon:
            exitinfo["reason"] = "converge"
            break

    return U_subset, iteration, losses, exitinfo
