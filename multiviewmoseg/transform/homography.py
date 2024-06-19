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
        assert data.shape[0] >= self.p_size
        src, dst = data[:, 0], data[:, 1]
        self.model.estimate(src=src, dst=dst)
        self.params = self.model.params

    def transform(self, src: np.ndarray) -> np.ndarray:
        assert self.is_fitted, "This transform does not have parameters"
        return homo2inhomo(inhomo2homo(src) @ self.params.T)

    def inverse_transform(self, dst: np.ndarray) -> np.ndarray:
        assert self.is_fitted, "This transform does not have parameters"
        return homo2inhomo(np.linalg.solve(self.params, inhomo2homo(dst).T))

    def residuals(self, data: np.ndarray):
        src, dst = data[:, 0], data[:, 1]
        return homography_residual(src=src, dst=dst, H=self.params)

    def is_degenerate(data: np.ndarray):
        src, dst = data[:, 0], data[:, 1]
        return is_colinear_2d_inhomo(src) or is_colinear_2d_inhomo(dst)


class MultiFrameHomographyTransform(Transform):
    p_size = 4
    n_frames: int

    def __init__(
        self,
        params: np.ndarray = None,
        n_frames: int = None,
    ) -> None:
        assert n_frames > 1
        super().__init__(params)

        if params is not None:
            self.n_frames = params.shape[0] + 1
        else:
            self.n_frames = n_frames

    def fit(self, data: np.ndarray):
        """data should be of shape (N, n_frames, 2)"""
        assert data.shape[0] >= self.p_size
        assert data.shape[1] == self.n_frames
        assert data.shape[2] == 2
        params = []

        model = ProjectiveTransform()
        for i in range(self.n_frames - 1):
            model.estimate(
                src=data[:, i, :],
                dst=data[:, i + 1, :],
            )
            params.append(self.model.params)

        self.params = np.array(params)

    def residuals(self, data: np.ndarray, mode="mean"):
        assert data.shape[1] == self.n_frames
        assert data.shape[2] == 2
        residuals = []
        for i in range(self.n_frames - 1):
            residuals.append(
                homography_residual(
                    src=data[:, i, :],
                    dst=data[:, i + 1, :],
                    H=self.params[i],
                )
            )

        if mode == "mean":
            return np.array(residuals).mean(axis=0)
        elif mode == "max":
            return np.array(residuals).max(axis=0)
        else:
            raise NotImplementedError

    def is_degenerate(data: np.ndarray):
        return any([is_colinear_2d_inhomo(data[:, i, :]) for i in range(data.shape[1])])
