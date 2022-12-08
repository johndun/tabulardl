"""Array of categorical feature classes."""
from dataclasses import dataclass
from itertools import chain

import numpy as np

from .base import DataType
from .categorical import CategoricalFeature


@dataclass
class CategoricalArrayFeature(CategoricalFeature):
    """A feature consisting of an array of categorical features.

    Arrays need not be the same size and will be padded or truncated during transform.

    Parameters:
        max_len: Maximum sequence length. Long sequences will be right-truncated.
        pad_value: Pad value to use to increase short sequences to `max_len`.

    """
    type: DataType = DataType.CATEGORICAL_ARRAY
    max_len: int = 10
    pad_value: str = '<PAD>'

    def _fit_data_transformer(self, data):
        self.value_map = {self.pad_value: 0}
        if self.missing_value != self.pad_value:
            self.value_map[self.missing_value] = max(self.value_map.values()) + 1
        if self.unknown_value != self.pad_value:
            self.value_map[self.unknown_value] = max(self.value_map.values()) + 1
        self._fit(chain(*[x for x in data if x]))

    def transform_single(self, data):
        data = data if data is not None else []
        data = [data[idx] if idx < len(data) else self.pad_value for idx in range(self.max_len)]
        return np.array([
            self.value_map[
                x if x in self.value_map else
                self.missing_value if x is None else
                self.unknown_value
            ]
            for x in (data or [])
        ])

    def __repr__(self):
        key_repr = (
            f'{self.key}'
            if isinstance(self.key, list) else
            '"' + self.key + '"'
        )
        string = [
            'CategoricalArrayFeature(',
            f'    id="{self.id}"',
            f'    key={key_repr}',
        ]
        if self.is_fit:
            string += [
                f'    dict_size={self.dictionary_size}',
                f'    seq_len={self.max_len}',
                f'    dict={list(self.value_map.items())[:5]}',
            ]
        string += [')']
        return '\n'.join(string)
