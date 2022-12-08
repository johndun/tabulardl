"""Basic tabular or columnar dataset comprised of named torch Tensors."""
from dataclasses import dataclass
from typing import Dict, Union, Optional

import torch
from torch.utils.data import DataLoader


@dataclass
class TabularDataset:
    """A tabular dataset of named torch tensors.

    Each tensor should have identical size in the 1st dimension.

    Parameters:
        data: A dictionary of tensors with sample indexed in the first dimension.

    """
    data: Dict[str, torch.Tensor]

    def __getitem__(self, idx):
        return {key: data[idx] for key, data in self.data.items()}

    def __len__(self):
        for data in self.data.values():
            return len(data)

    def join(self, other: 'TabularDataset') -> 'TabularDataset':
        """Joins two identically structured `TabularDataset` instances.

        Args:
            other: Dataset to join with `self`.

        Returns:
            A joined (e.g., unioned in SQL terms) `TabularDataset`.

        """
        joined = {}
        for key, datum in self.data.items():
            joined[key] = torch.cat((datum, other.data[key]), dim=0)  # pylint: disable=[E1101]
        return TabularDataset(data=joined)

    def loader(self, **kwargs):
        return DataLoader(self, **kwargs)

    @staticmethod
    def load_from(path):
        return TabularDataset(torch.load(path))

    def save_to(self, path):
        torch.save(self.data, path)
