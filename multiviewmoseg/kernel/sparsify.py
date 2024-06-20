import numpy as np


def adapt_enn(A: np.ndarray, alpha: float = 0.5):
    """
    Function to adaptively e-nn sparsify affinity matrix.
    A: affinity matrix, shape (N, N)
    """
    signma: np.ndarray = (A**alpha).max(axis=1, keepdims=True) - A**alpha  # (N, N)
    prob = signma / signma.sum(axis=1, keepdims=True)  # (N, N)

    E = (prob * np.log(prob + np.finfo(float).eps)).sum(axis=1, keepdims=True)  # (N, 1)

    sparsity_mask = (np.log(prob + np.finfo(float).eps) < E).astype(float)
    sparsity_mask = (sparsity_mask + sparsity_mask.T) / 2

    A_sparse = sparsity_mask * A

    return A_sparse
