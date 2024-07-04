import numpy as np
from typing import Union, List


def coregularize_loss(U_coreg: np.ndarray, L_coreg: np.ndarray):
    return sum([np.trace(U_coreg[i].T @ L_coreg[i] @ U_coreg[i]) for i in range(U_coreg.shape[0])])


def coregularize_eig(
    kernels: Union[np.ndarray, List[np.ndarray]],
    num_motion: int,
    lambda_: float,
    epsilon: float = 1e-8,
    max_iter: int = 30,
):
    """
    Input:
        kernels (list of np.ndarray): List of kernel matrices, of shape (V, N, N), v is the number of kernels
        num_motion (int): Number of motions
        lambda_ (float): Co-regularization parameter
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
        K_i = kernels[i]  # (N, N)
        D_i = np.sqrt(np.sum(K_i, axis=1, keepdims=True)) + 1e-8

        L_i = np.eye(D_i.shape[0]) - K_i / D_i / D_i.T
        L.append(L_i)

        U_tmp, _, _ = np.linalg.svd(L_i)
        U_i = U_tmp[:, -num_motion:]
        U.append(U_i)

    # start Co-Regularization
    U_coreg = np.array(U)  # (v, N, num_motion)
    L_coreg = np.array(L)  # (v, N, N)

    losses = [coregularize_loss(U_coreg=U_coreg, L_coreg=L_coreg)]

    exitinfo = {"reason": "timeout"}

    # initialize U@U.T as a intermediate variable
    UUt = np.array([U_coreg[i] @ U_coreg[i].T for i in range(num_kernels)])  # (v, N, N)

    for iteration in range(max_iter):

        for i in range(num_kernels):
            UUt_i = UUt.sum(axis=0) - U_coreg[i] @ U_coreg[i].T
            L_coreg[i] = L[i] - lambda_ * UUt_i

            D_i, vec_i = np.linalg.eigh((L_coreg[i] + L_coreg[i].T) / 2)
            idx = np.argsort(D_i)

            U_coreg[i] = vec_i[:, idx[:num_motion]]

            UUt[i] = U_coreg[i] @ U_coreg[i].T

        losses.append(coregularize_loss(U_coreg=U_coreg, L_coreg=L_coreg))

        if abs(losses[-2] - losses[-1]) < epsilon:
            exitinfo["reason"] = "converge"
            break

    return U_coreg, iteration, losses, exitinfo
