from typing import Any

import numpy as np


class Transform:
    """A base model for Affine, Homography and Fundamental matrix transform."""

    params: np.ndarray = None
    p_size: int  # mininum number of data needed to fit this model

    def __init__(self, params=None) -> None:
        self.params = params

    @property
    def is_fitted(self) -> bool:
        return self.params is not None

    def transform(self, src: np.ndarray) -> Any:
        raise NotImplementedError

    def inverse_transform(self, dst: np.ndarray) -> Any:
        raise NotImplementedError

    def is_degenerate(data: np.ndarray):
        """Given input fitting data, test if the data generates degenerate model parameters"""
        raise NotImplementedError

    def fit(self, data: np.ndarray):
        """Given input data, find the set of parameters that fits the data"""
        raise NotImplementedError

    def residuals(self, data: np.ndarray):
        """Given the input points, return the model fitting distance, or equivalently, the redisuals."""
        raise NotImplementedError
