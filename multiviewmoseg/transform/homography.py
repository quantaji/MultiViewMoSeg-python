import numpy as np
from skimage.transform._geometric import ProjectiveTransform

from .base import Transform
from .utils import homo2inhomo, inhomo2homo, homography_residual, is_colinear_2d_inhomo


class HomographyTransform(Transform):
    p_size = 4

    def __init__(self) -> None:
        super().__init__()
        self.model = ProjectiveTransform(matrix=self.params)

    def fit(self, data: np.ndarray):
        """x1 and x2: shape (N, 2) in-homogeneous representation of 2D points. Code adapted from"""
        src, dst = data[:, :2], data[:, 2:]
        self.model.estimate(src=src, dst=dst)
        self.params = self.model.params

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
