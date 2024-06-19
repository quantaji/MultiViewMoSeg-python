import numpy as np


def compute_ordered_redisual_kernel(
    residual_1: np.ndarray,
    residual_2: np.ndarray,
    percentile: float = 0.1,
):
    """
    residual_1, residual_2 is of shape (num_hypo, num_data_1), (num_hypo, num_data_2)

    return:
        K: of shape (num_data_1, num_data_2)
    """

    assert residual_1.shape[0] == residual_2.shape[0]

    datawise_percentile_1 = np.percentile(residual_1, percentile * 100, axis=0)
    datawise_percentile_2 = np.percentile(residual_2, percentile * 100, axis=0)

    selected_hypo_mask_1 = (residual_1 < datawise_percentile_1.reshape(1, -1)).astype(float)
    selected_hypo_mask_2 = (residual_2 < datawise_percentile_2.reshape(1, -1)).astype(float)

    K = selected_hypo_mask_1.T @ selected_hypo_mask_2

    return K
