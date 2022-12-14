"""Numeric feature classes."""
from dataclasses import dataclass, field
from typing import Optional, List, Union

import numpy as np
import torch

from .base import Feature, DataType


@dataclass
class NumericFeature(Feature):  # pylint: disable=[R0902]
    """A numeric, possibly vector-valued feature.

    Supports 1 and 2 dimension features.

    Parameters:
        missing_value: Impute missing values with this.
        center: If true, apply centering transformation so that the average transformed value among
            the data provided to `fit_data_transformer` is 0.
        scale: If true, apply scaling transformation so that the standard deviation of transformed
            values in the data provided to `fit_data_transformer` is 1.
        unit_scale: If true, rescales to [0, 1] using `clip_percentiles`. `center` and `scale`
            parameters are ignored in this case.
        clip_percentiles: An array of (lower, upper) percentiles at which to clip raw data values.
            If None, no clipping is performed.
        mean: A float (for 1 dim features) or numpy array (for 2 dim features) containing means.
        std: A float (for 1 dim features) or numpy array (for 2 dim features) containing standard
            deviations.
        clip_values: A numpy array containing clipping values.

    """
    type: DataType = DataType.NUMERIC
    missing_value: Optional[float] = None
    center: bool = False
    scale: bool = False
    unit_scale: bool = False
    clip_percentiles: List[Union[None, float]] = field(default_factory=lambda: [None, None])
    mean: Union[float, np.ndarray] = None
    std: Union[float, np.ndarray] = None
    clip_values: np.ndarray = None

    def _fit_data_transformer(self, data):
        data = (
            # Replace None values with self.missing_value
            [x if x is not None else self.missing_value for x in data]
            if self.missing_value is not None else
            # Or drop None values is no missing_value is set
            [x for x in data if x is not None]
        )
        if len(data):
            self.mean = self.mean if self.mean is not None else np.mean(data, axis=0)
            self.std = self.std if self.std is not None else np.std(data, axis=0)
            self.clip_values = self.clip_values if self.clip_values is not None else np.percentile(
                data,
                q=[
                    100 * (self.clip_percentiles[0] or 0.),
                    100 * (self.clip_percentiles[1] or 1.)
                ],
                axis=0
            )

    def _transform_raw_data(self, data):
        lower, upper = self.clip_values
        missing_value = self.missing_value if self.missing_value is not None else self.mean
        if not isinstance(data, (list, np.ndarray)):
            data = [data]
        data = np.array(data).astype(np.float32)
        data = np.nan_to_num(data, nan=missing_value)
        if (
                self.clip_percentiles[0] is not None
                or self.clip_percentiles[1] is not None
        ):
            data = np.clip(
                data,
                a_min=lower if self.clip_percentiles[0] is not None else None,
                a_max=upper if self.clip_percentiles[1] is not None else None
            )
        if self.unit_scale:
            data = (data - lower) / (upper - lower)
        else:
            if self.center:
                data -= self.mean
            if self.scale:
                std = self.std
                if len(std.shape) == 0 and std < 0.001:
                    std = 1.
                elif len(std.shape):
                    std[std < 0.001] = 1.
                data /= std
        data = torch.FloatTensor(data)  # pylint: disable=[E1101]
        if len(data.shape) == 1:
            data = data.unsqueeze(-1)
        return data

    def __repr__(self):
        key_repr = (
            f'{self.key}'
            if isinstance(self.key, list) else
            "'" + self.key + "'"
        )
        string = [
            'NumericFeature(',
            f'    id="{self.id}"',
            f'    key={key_repr}',
            f'    shape={self.mean.shape}',
            f'    missing_value={self.missing_value}',
            f'    center={self.center}',
            f'    scale={self.scale}',
            f'    unit_scale={self.unit_scale}',
            f'    clip_percentiles={self.clip_percentiles}',
        ]
        if len(self.mean.shape) == 0:
            string += [
                f'    mean={self.mean}',
                f'    std={self.std}',
                f'    clip_values={self.clip_values}',
            ]
        string += [')']
        return '\n'.join(string)
