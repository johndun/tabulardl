from itertools import islice
from typing import Iterable, List

import numpy as np


def is_numeric(data):
    def _is_float(element: any) -> bool:
        if not element:
            return True
        if isinstance(element, list):
            return False
        try:
            float(element)
            return True
        except ValueError:
            return False

    def _is_int(element: any) -> bool:
        if not element:
            return True
        if float(element) == int(float(element)):
            return True
        return False

    if np.mean([_is_float(x) for x in data]) == 1:
        if np.mean([_is_int(x) for x in data]) == 1:
            return 'int'
        return 'float'
    return 'non-numeric'


def convert_raw_data(data):
    datatype = is_numeric(data)
    if datatype in('int', 'float'):
        data = [x if len(x) else None for x in data]
        if datatype == 'float':
            return [float(x) if x is not None else None for x in data]
        return [int(x) if x is not None else None for x in data]
    return data


def chunked_iterator(iterable: Iterable, chunk_size: int) -> List:
    """An iterator that yields chunks.

    No samples are discarded. The final chunk may be smaller.

    Args:
        iterable: An iterable.
        chunk_size: The number of samples from `iterable` to return each yield.

    Returns: A list of items from `iterable`
    """
    it = iter(iterable)
    while True:
        chunk = list(islice(it, chunk_size))
        if not chunk:
            return
        yield chunk
