import numpy as np
from skimage.transform._geometric import ProjectiveTransform

from .base import Transform
from .utils import homo2inhomo, inhomo2homo, homography_residual, is_colinear_2d_inhomo


class HomographyTransform(Transform):

    def __init__(self) -> None:
        super().__init__()
        self.model = ProjectiveTransform()

    def fit(self, src: np.ndarray, dst: np.ndarray):
        """x1 and x2: shape (N, 2) in-homogeneous representation of 2D points. Code adapted from"""

        self.model.estimate(src=src, dst=dst)
        self.params = self.model.params
        self.is_fitted = True

    def transform(self, src: np.ndarray) -> np.ndarray:
        assert self.is_fitted, "This transform does not have parameters"
        return homo2inhomo(inhomo2homo(src) @ self.params.T)

    def inverse_transform(self, dst: np.ndarray) -> np.ndarray:
        assert self.is_fitted, "This transform does not have parameters"
        return homo2inhomo(np.linalg.solve(self.params, inhomo2homo(dst).T))

    def residuals(self, src: np.ndarray, dst: np.ndarray):
        return homography_residual(src=src, dst=dst, H=self.params)

    def is_degenerate(self, src: np.ndarray, dst: np.ndarray):
        return is_colinear_2d_inhomo(src) or is_colinear_2d_inhomo(dst)
