from .base import Sampler
from ..transform.base import Transform
import numpy as np
import warnings


class RandomSampler(Sampler):

    def __init__(
        self,
        max_degen_trials: int = 10,  # max number of inner loop to avoid degeneracy
    ) -> None:
        super().__init__()
        self.max_degen_trials = max_degen_trials

    def sample(
        self,
        num_hypo: int,
        transform: Transform,
        data: np.ndarray,
    ):
        """data is of shape (N, d), where N is the number of data."""
        data_size = data.shape[0]
        assert transform.p_size is not None
        assert data_size >= transform.p_size
        index_list = np.arange(data_size)

        hypos_params = []  # of shape (num_hypo, d1, d2, ...)
        hypos_residuals = []  # of shape (num_hypo, num_data)
        hypos_indices = []  # of shape (num_hypo, p_size)

        for i in range(num_hypo):

            degen_count = 0
            is_degen = True
            p_subset = None

            for j in range(self.max_degen_trials):
                degen_count += 1

                sampled_indices = np.random.choice(
                    a=index_list,
                    size=transform.p_size,
                    replace=False,
                )
                p_subset = data[sampled_indices]

                is_degen = transform.is_degenerate(data=p_subset)

                if not is_degen:
                    break

            if is_degen:
                warnings.warn("Cannot find a valid p-subset!")
                continue

            # use the sampled data to fit a model
            hypo = transform()
            hypo.fit(data=p_subset)

            # calculate the fitting error of all data to this model
            residuals = hypo.residuals(data=data)

            hypos_params.append(hypo.params)
            hypos_residuals.append(residuals)
            hypos_indices.append(sampled_indices)

        hypos_params = np.array(hypos_params)
        hypos_residuals = np.array(hypos_residuals)
        hypos_indices = np.array(hypos_indices)

        return hypos_params, hypos_residuals, hypos_indices
