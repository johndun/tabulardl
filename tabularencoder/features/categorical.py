"""Categorical feature classes."""
from dataclasses import dataclass
from typing import Dict, Optional, Union, List

import numpy as np
import torch

from .base import Feature, DataType


class ValueNotFoundError(Exception):
    """Exception for initializing a CategoricalFeature with a dictionary with missing keys."""


@dataclass
class CategoricalFeature(Feature):
    """A categorical feature using integer value encoding.

    Parameters:
        max_vocab_size: Maximum vocabulary dictionary size. The most frequently occurring raw
            values will have keys. Anything else will be replaced with `unknown_value`.
        low_count_threshold: Minimum occurrence count for a raw value to recieve a key.
        unknown_value: String or integer value to use to replace unknown raw values. Can be the
            same as `missing_value`.
        missing_value: String or integer value to use to replace missing (None) raw values. Can
            be the same as `unknown_value`.
        value_map: Mapping between raw data and integer values where larger values indicate the
            larger number of occurrences for the associated raw data value in the data set used
            to initialize this parameters.

    """
    type: DataType = DataType.CATEGORICAL
    max_vocab_size: int = 10000
    low_count_threshold: Optional[int] = 0
    unknown_value: Union[str, int] = '<UNKNOWN>'
    missing_value: Union[str, int] = '<MISSING>'
    value_map: Dict = None

    def __post_init__(self):
        if self.value_map:
            if (
                    self.missing_value not in self.value_map
                    or self.missing_value not in self.value_map
            ):
                raise ValueNotFoundError
            self.is_fit = True

    @property
    def dictionary_size(self):
        """Returns the number of items in the vocabulary dictionary."""
        return len(self.value_map)

    def _fit(self, data: List):
        """Generates a vocabulary dictionary from a training dataset."""
        value_counts = {}
        for val in data:
            if val:
                value_counts[val] = value_counts.get(val, 0) + 1
        idx = max(self.value_map.values()) + 1
        for key, count in sorted(value_counts.items(), key=lambda x: -x[1]):
            if idx >= self.max_vocab_size:
                break
            if count >= self.low_count_threshold:
                self.value_map[key] = idx
                idx += 1

    def _fit_data_transformer(self, data):
        self.value_map = {self.missing_value: 0}
        if self.unknown_value != self.missing_value:
            self.value_map[self.unknown_value] = 1
        self._fit(data)

    def transform_single(self, data: Union[str, int]):
        """Transform a single value."""
        return self.value_map[
            data if data in self.value_map else
            self.missing_value if data is None else
            self.unknown_value
        ]

    def _transform_raw_data(self, data):
        if not isinstance(data, list):
            data = [data]
        # pylint: disable=[E1101]
        return torch.LongTensor(np.array([self.transform_single(y) for y in data]))

    def __repr__(self):
        key_repr = (
            f'{self.key}'
            if isinstance(self.key, list) else
            '"' + self.key + '"'
        )
        string = [
            'CategoricalFeature(',
            f'    id="{self.id}"',
            f'    key={key_repr}',
        ]
        if self.is_fit:
            string += [
                f'    dict_size={self.dictionary_size}',
                f'    dict={list(self.value_map.items())[:5]}',
            ]
        string += [')']
        return '\n'.join(string)
