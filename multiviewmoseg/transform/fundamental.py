import numpy as np
from skimage.transform._geometric import ProjectiveTransform, FundamentalMatrixTransform as FMT, _center_and_normalize_points

from .base import Transform
from .utils import homo2inhomo, inhomo2homo, homography_residual, is_colinear_2d_inhomo


class FundamentalMatrixTransform(Transform):

    def __init__(self) -> None:
        super().__init__()
        self.model = FMT()

    def fit(self, src: np.ndarray, dst: np.ndarray):
        """x1 and x2: shape (N, 2) in-homogeneous representation of 2D points. Code adapted from"""
        try:
            self.model.estimate(src=src, dst=dst)
            self.params = self.model.params
            self.is_fitted = True
        except:
            self.is_fitted = False

    def residuals(self, src: np.ndarray, dst: np.ndarray):
        assert self.is_fitted, "This transform does not have parameters"
        return self.model.residuals(src=src, dst=dst)

    def is_degenerate(self, src: np.ndarray, dst: np.ndarray):
        src = np.asarray(src)
        dst = np.asarray(dst)
        if src.shape != dst.shape:
            raise ValueError("src and dst shapes must be identical.")
        if src.shape[0] < 8:
            return True

        # Center and normalize image points for better numerical stability.
        try:
            src_matrix, src = _center_and_normalize_points(src)
            dst_matrix, dst = _center_and_normalize_points(dst)
        except ZeroDivisionError:
            return True

        A = np.ones((src.shape[0], 9))
        A[:, :2] = src
        A[:, :3] *= dst[:, 0, np.newaxis]
        A[:, 3:5] = src
        A[:, 3:6] *= dst[:, 1, np.newaxis]
        A[:, 6:8] = src

        rank = np.linalg.matrix_rank(A)

        return rank < 8
