from typing import Any

import numpy as np


class Transform:
    """A base model for Affine, Homography and Fundamental matrix transform."""

    params: np.ndarray = None
    is_fitted: bool = False

    def transform(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError

    def inverse_transform(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError

    def is_degenerate(self, x):
        """Given input fitting data, test if the data generates degenerate model parameters"""
        raise NotImplementedError

    def fit(self, x):
        """Given input data, find the set of parameters that fits the data"""
        raise NotImplementedError

    def residuals(self, x):
        """Given the input points, return the model fitting distance, or equivalently, the redisuals."""
        raise NotImplementedError
