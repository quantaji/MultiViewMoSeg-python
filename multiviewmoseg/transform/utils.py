import numpy as np


def inhomo2homo(x: np.ndarray):
    """Assumes that the last dimention is the data demention"""
    ones = np.ones(x.shape[:-1] + (1,))
    return np.concatenate([x, ones], axis=-1)


def homo2inhomo(x: np.ndarray):
    """Assumes that the last dimention is the data demention"""
    return x[..., :-1] / x[..., -1:]


def normalize_homo(x: np.ndarray):
    """Normalises array of homogeneous coordinates to a scale of 1

    Note that any homogeneous coordinates at infinity (having a scale value of 0) are left unchanged.

    x: shape (..., d+1)
    """
    finite_mask = np.abs(x[..., -1]) > np.finfo(float).eps * 1e4

    if (~finite_mask).sum() > 0:
        print("Warning: Some points are at infinity")

    new_x = x.copy()
    new_x[finite_mask] /= new_x[finite_mask, -1:]

    return new_x


def homography_residual(src: np.ndarray, dst: np.ndarray, H: np.ndarray):
    """src, dst: np.ndarray of shape (..., d+1), H: (d+1, d+1)"""

    src_norm = normalize_homo(inhomo2homo(src))
    trans_src_norm = normalize_homo(inhomo2homo(src) @ H.T)

    dst_norm = normalize_homo(inhomo2homo(dst))  # (..., d+1)
    inv_trans_dst_norm = normalize_homo(np.linalg.solve(H, inhomo2homo(dst).T).T)  # (..., d+1)

    dist = np.sum((src_norm - inv_trans_dst_norm) ** 2, axis=-1) + np.sum((dst_norm - trans_src_norm) ** 2, axis=-1)  # (...)

    return dist


def is_colinear_2d_inhomo(x: np.ndarray):
    """x: shape of (N, 2)"""
    direction_vec = np.diff(x, axis=0)  # (N-1, 2)
    cross_residual = direction_vec[:-1, 0] * direction_vec[1:, 1] - direction_vec[:-1, 1] * direction_vec[1:, 0]
    cross_residual = cross_residual.max()

    return cross_residual < np.finfo(float).eps * 1e4
