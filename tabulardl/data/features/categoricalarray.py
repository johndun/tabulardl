"""Array of categorical feature classes."""
from dataclasses import dataclass
from itertools import chain

import numpy as np

from tabulardl.data.features.base import DataType
from tabulardl.data.features.categorical import CategoricalFeature


@dataclass
class CategoricalArrayFeature(CategoricalFeature):
    """A feature consisting of an array of categorical features.

    Arrays need not be the same size and will be padded or truncated during transform.

    Parameters:
        max_len: Maximum sequence length. Long sequences will be right-truncated.
        pad_value: Pad value to use to increase short sequences to `max_len`.

    """
    data_type: DataType = DataType.CATEGORICAL_ARRAY
    max_len: int = 10
    pad_value: str = '<PAD>'

    def _fit_data_transformer(self, data):
        self.value_map = {self.missing_value: 0}
        if self.unknown_value != self.missing_value:
            self.value_map[self.unknown_value] = max(self.value_map.values()) + 1
        if self.pad_value != self.missing_value:
            self.value_map[self.pad_value] = max(self.value_map.values()) + 1
        self._fit(chain(*data))

    def transform_single(self, data):
        data = [data[idx] if idx < len(data) else self.pad_value for idx in range(self.max_len)]
        return np.array([
            self.value_map[
                x if x in self.value_map else
                self.missing_value if data is None else
                self.unknown_value
            ]
            for x in data
        ])
