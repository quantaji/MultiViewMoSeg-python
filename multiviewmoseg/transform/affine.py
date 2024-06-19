import numpy as np

from .base import Transform
from .utils import homo2inhomo, inhomo2homo, homography_residual, is_colinear_2d_inhomo


def affine_transform_fit(src: np.ndarray, dst: np.ndarray):
    """src and dst: shape (N, 2) in-homogeneous representation of 2D points. Code adapted from vgg_Haffine_from_x_MLE.
    returns a 3x3 matrix A"""

    # normalization for src and dst points
    mean_src = np.mean(src, axis=0)
    max_std_src = np.max(np.std(src, axis=0))

    condition_src = np.diag([1 / max_std_src, 1 / max_std_src, 1])
    condition_src[:2, 2] = -mean_src / max_std_src

    mean_dst = np.mean(dst, axis=0)
    max_std_dst = np.max(np.std(dst, axis=0))

    condition_dst = np.diag([1 / max_std_dst, 1 / max_std_dst, 1])
    condition_dst[:2, 2] = -mean_dst / max_std_dst

    src_norm_homo = inhomo2homo(src) @ condition_src.T
    dst_norm_homo = inhomo2homo(dst) @ condition_dst.T

    src_norm = homo2inhomo(src_norm_homo)  # (N, 2)
    dst_norm = homo2inhomo(dst_norm_homo)  # (N, 2)

    A = np.hstack([src_norm, dst_norm])  # (N, 4)

    U, S, Vt = np.linalg.svd(A)  # U (N, N); S min(N, 4); V (4, 4)

    nullspace_dimension = np.sum(S < np.finfo(float).eps * S[1] * 1e3)
    if nullspace_dimension > 2:
        print("Nullspace is a bit roomy...")

    V = Vt.T
    B = V[:2, :2]
    C = V[2:4, :2]

    H = np.vstack([np.hstack([C @ np.linalg.pinv(B), np.zeros((2, 1))]), [0, 0, 1]])

    # decondition
    H = np.linalg.inv(condition_dst) @ H @ condition_src

    return H / H[2, 2]


class AffineTransform(Transform):
    p_size = 3

    def fit(self, data: np.ndarray):
        """x1 and x2: shape (N, 2) in-homogeneous representation of 2D points. Code adapted from"""
        src, dst = data[:, :2], data[:, 2:]
        assert src.shape[0] >= 3
        self.params = affine_transform_fit(src=src, dst=dst)

    def transform(self, src: np.ndarray) -> np.ndarray:
        assert self.is_fitted, "This transform does not have parameters"

        return homo2inhomo(inhomo2homo(src) @ self.params.T)

    def inverse_transform(self, dst: np.ndarray) -> np.ndarray:
        assert self.is_fitted, "This transform does not have parameters"

        return homo2inhomo(np.linalg.solve(self.params, inhomo2homo(dst).T))

    def residuals(self, data: np.ndarray):
        src, dst = data[:, :2], data[:, 2:]
        return homography_residual(src=src, dst=dst, H=self.params)

    def is_degenerate(data: np.ndarray):
        src, dst = data[:, :2], data[:, 2:]
        return is_colinear_2d_inhomo(src) or is_colinear_2d_inhomo(dst)
