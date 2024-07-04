import numpy as np

from ..transform.base import Transform


class Sampler:

    def sample(
        num_hypo: int,
        transform: Transform,
        data: np.ndarray,
    ):
        raise NotImplementedError
