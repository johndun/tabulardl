"""Constants and abstract feature classes."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Union

import torch


class DataType(Enum):
    NUMERIC = 'numeric'
    CATEGORICAL = 'categorical'
    CATEGORICAL_ARRAY = 'categorical_array'


class FeatureTransformBeforeFitError(Exception):
    """Exception raised when a Feature transform method is called before the fit method."""


@dataclass
class Feature:
    """Base class for a dataset column transformer.

    Parameters:
        data_type: DataType object that describes the type of data.
        is_fit: If not true, `transform_raw_data` method cannot be used.
    """
    id: str
    key: Union[str, List]
    type: DataType
    is_fit: bool = False

    def _fit_data_transformer(self, data: List):
        """Using provided data, instantiate state required to convert raw data to vectorized form.

        Must be overridden by subclasses.

        Args:
            data: A list of raw data.

        """
        raise NotImplementedError

    def _transform_raw_data(self, data: List) -> torch.Tensor:
        """Transforms raw data.

        Must be overridden by subclasses.

        Args:
            data: A list of raw data.

        Returns:
            A `torch` tensor containing numeric, transformed data.
        """
        raise NotImplementedError

    def fit_data_transformer(self, data: List):
        """Calls `_fit_data_transformer` and sets `is_fit=True`."""
        self._fit_data_transformer(data)
        self.is_fit = True

    def transform_raw_data(self, data: List) -> torch.Tensor:
        """Checks if data transformer is fitted and calls `_transform_raw_data`."""
        if not self.is_fit:
            raise FeatureTransformBeforeFitError
        return self._transform_raw_data(data)
