"""Classes that provide access (e.g., index retrieval, mini batch generators) to data sets."""

from dataclasses import dataclass
from typing import Dict, Union

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, BatchSampler


@dataclass
class TabularDataset:
    """A tabular dataset of named `torch` tensors allowing for nested keys.

    Each tensor should have identical shapes in the 1st dimension.

    Useful for RL use-cases where samples may have nested state and next-state inputs.

    Parameters:
        data: A dictionary (with possibly nested string keys) of tensors with sample indexed in the
            first dimension.

    """
    data: Dict[str, Union[torch.Tensor, 'TabularDataset']]

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
            if isinstance(datum, torch.Tensor):
                joined[key] = torch.cat((datum, other.data[key]), dim=0)  # pylint: disable=[E1101]
            else:
                joined[key] = datum.join(other.data[key])
        return TabularDataset(data=joined)

    def loader(
            self,
            batch_size: int,
            shuffle: bool = True,
            drop_last: bool = False,
            pin_memory: bool = False
    ) -> DataLoader:
        """Returns a `torch` data loader.

        Args:
            batch_size: Mini batch size.
            shuffle: If true, data will be shuffled.
            drop_last: If true, final mini batches of size not equal to `batch_size` will be
                dropped.
            pin_memory: If true, uses `torch` memory pinning.

        Returns:
            A data loader that can be used to generate mini batches for model training and
                evaluation.

        """
        return DataLoader(
            dataset=self,
            sampler=BatchSampler(
                RandomSampler(self) if shuffle else SequentialSampler(self),
                batch_size=batch_size,
                drop_last=drop_last
            ),
            collate_fn=lambda x: x[0],
            pin_memory=pin_memory
        )
