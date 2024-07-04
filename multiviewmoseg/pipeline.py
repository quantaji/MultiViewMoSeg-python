import numpy as np
from typing import Dict

from .kernel import compute_ordered_redisual_kernel, adapt_enn_sparsify
from .sampler import Sampler
from .transform import AffineTransform, FundamentalMatrixTransform, HomographyTransform
from .moseg import subset_eig


def get_kernels_from_data(
    trajectory_data: np.ndarray,
    sampler: Sampler,
    num_hypo: int,
    visibility_data: np.ndarray = None,
    mode: str = "AHF",
):
    """
    trajectory_data: (N, T, 2)
    visibility_data: (N, T)
    returns a dictionary of kernel
    """

    kernels = {}
    num_frames = trajectory_data.shape[1]
    num_data = trajectory_data.shape[0]

    transform_map = {
        "A": AffineTransform,
        "H": HomographyTransform,
        "F": FundamentalMatrixTransform,
    }

    for transform_label in ["A", "H", "F"]:
        if transform_label in mode:
            kernel = np.zeros(shape=(num_data, num_data))
            for i in range(num_frames - 1):

                if visibility_data is not None:
                    visible_mask = visibility_data[:, i] * visibility_data[:, i + 1]
                else:
                    visible_mask = np.ones_like(trajectory_data[:, 0, 0], dtype=bool)

                kernel_vis_mask = visible_mask.reshape(-1, 1) + visible_mask.reshape(1, -1)

                data = trajectory_data[visible_mask, i : i + 2]

                try:
                    _, residuals, _ = sampler.sample(
                        num_hypo=num_hypo,
                        transform=transform_map[transform_label],
                        data=data,
                    )

                    ORK = compute_ordered_redisual_kernel(residuals)

                    kernel[kernel_vis_mask] += ORK.reshape(-1)
                except:
                    pass

            # normalize by visibility count
            if visibility_data is not None:
                vis_count_normalizer = visibility_data @ visibility_data.T + 0.1
            else:
                vis_count_normalizer = num_frames * num_frames
            kernel = kernel / vis_count_normalizer

            # add to result
            kernels[transform_label] = kernel

    return kernels


def subset_constrained_clustering(
    kernels: Dict,
    num_motion: int,
    alpha: float,
    gamma: float = 1e-2,
):
    """
    alpha:  power scaling parameter
    """
    kernels_list = [adapt_enn_sparsify(kernels[label], alpha=alpha) for label in ["A", "H", "F"]]

    U_subset, iteration, losses, exitinfo = subset_eig(
        kernels=kernels_list,
        num_motion=num_motion,
        gamma=gamma,
    )

    return U_subset, iteration, losses, exitinfo

# TODO: single model spectrum clustring: A, H, F
# TODO: Kernel addition spectrum clustering
# TODO: co-recularization
