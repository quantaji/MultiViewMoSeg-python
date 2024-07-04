import numpy as np
from skimage.transform._geometric import ProjectiveTransform, FundamentalMatrixTransform as FMT, _center_and_normalize_points

from .base import Transform
from .utils import homo2inhomo, inhomo2homo, homography_residual, is_colinear_2d_inhomo


def fundamental_matrix_residuals(src: np.ndarray, dst: np.ndarray, F: np.ndarray):
    """copied from skimage"""
    src_homogeneous = np.column_stack([src, np.ones(src.shape[0])])
    dst_homogeneous = np.column_stack([dst, np.ones(dst.shape[0])])

    F_src = F @ src_homogeneous.T
    Ft_dst = F.T @ dst_homogeneous.T

    dst_F_src = np.sum(dst_homogeneous * F_src.T, axis=1)

    return np.abs(dst_F_src) / np.sqrt(F_src[0] ** 2 + F_src[1] ** 2 + Ft_dst[0] ** 2 + Ft_dst[1] ** 2)


class FundamentalMatrixTransform(Transform):
    p_size = 8

    def fit(self, data: np.ndarray):
        """x1 and x2: shape (N, 2) in-homogeneous representation of 2D points. Code adapted from"""
        assert data.shape[0] >= self.p_size
        src, dst = data[:, 0], data[:, 1]
        try:
            model = FMT(matrix=self.params)
            model.estimate(src=src, dst=dst)
            self.params = model.params
        except:
            pass

    def residuals(self, data: np.ndarray):
        assert self.is_fitted, "This transform does not have parameters"
        src, dst = data[:, 0], data[:, 1]
        return fundamental_matrix_residuals(
            src=src,
            dst=dst,
            F=self.params,
        )

    def is_degenerate(data: np.ndarray):
        src, dst = data[:, 0], data[:, 1]

        if src.shape != dst.shape:
            raise ValueError("src and dst shapes must be identical.")
        if src.shape[0] < 8:
            return True

        # Center and normalize image points for better numerical stability.
        try:
            _, src = _center_and_normalize_points(src)
            _, dst = _center_and_normalize_points(dst)
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


class MultiFrameFundamentalMatrixTransform(Transform):

    p_size = 8
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
        assert data.shape[0] >= self.p_size
        assert data.shape[1] == self.n_frames
        assert data.shape[2] == 2

        try:
            params = []
            model = FMT()

            for i in range(self.n_frames - 1):
                model.estimate(
                    src=data[:, i, :],
                    dst=data[:, i + 1, :],
                )
                params.append(model.params)

            self.params = np.array(params)

        except:
            pass

    def residuals(self, data: np.ndarray, mode="mean"):
        assert data.shape[1] == self.n_frames
        assert data.shape[2] == 2
        assert self.params is not None
        residuals = []

        for i in range(self.n_frames - 1):
            residuals.append(
                fundamental_matrix_residuals(
                    src=data[:, i, :],
                    dst=data[:, i + 1, :],
                    F=self.params[i],
                )
            )

        if mode == "mean":
            return np.array(residuals).mean(axis=0)
        elif mode == "max":
            return np.array(residuals).max(axis=0)
        else:
            raise NotImplementedError

    def is_degenerate(data: np.ndarray):
        return any([FundamentalMatrixTransform.is_degenerate(data=data[:, i : i + 2]) for i in range(data.shape[1] - 1)])
