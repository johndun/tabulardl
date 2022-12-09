import os
import pickle
from dataclasses import dataclass
from typing import Dict, Tuple

from torch import nn
from torch.utils.data import DataLoader

from tabularencoder import DataType


class RawDataElement:
    """A typing class to represent a single raw data record represented as a JSON-like object.

    Keys should be strings and values can be nested RawData objects.

    self: Dict[str, Union[None, str, int, float, List[str], List[int], List[float], 'RawData']
    """


class TransposedRawData:
    """A collection of `RawDataElement` items in transposed representation.

    All keys should be strings. Values should be lists of items that are valid values for a
    `RawDataElement` object OR a nested `TransposedRawData` object OR None.
    Each value list should consist of only a single data type (or Nones).
    Each value list should be the same length.

    self: Dict[
        str,
        Union['TransposedRawData', List[str], List[int], List[float], List[List[str]], ...
    ]
    """
    def __getitem__(self, idx):
        pass


class BaseRawData:
    """An unprocessed dataset.

    Override with parameters corresponding to individual experiments. E.g.:
    * interaction: List[RawDataElement]
    * item: List[RawDataElement]
    """


class BaseDataSchema:
    """An object containing feature definitions.

    Override with parameters corresponding to feature definitions for individual experiments. E.g.:
    * interaction: Dict[str, Feature]
    * item: Dict[str, Feature]
    """

    def save_to(self, path: str):
        pickle.dump(self, open(f'{path}/data_schema.pickle', 'wb'))

    @staticmethod
    def load_from(path: str):
        return pickle.load(open(f'{path}/data_schema.pickle', 'rb'))

    @staticmethod
    def group_features_by_type(features: Dict) -> Dict[str, Dict]:
        grouped_features = {datatype.value: [] for datatype in DataType}
        for feature in features.values():
            grouped_features[feature.type.value].append(feature)
        return grouped_features


class BaseExperimentDataset:
    """An object containing experiments for a model training experiment.

    Override with parameters corresponding to feature definitions for individual experiments. E.g.:
    * interaction: SplitsDataset
    * item: TabularDataset
    """
    def loader(self, **kwargs) -> DataLoader:
        return DataLoader(self, **kwargs)

    @staticmethod
    def load_from(path: str) -> 'BaseExperimentDataset':
        raise NotImplementedError

    def save_to(self, path: str):
        raise NotImplementedError


@dataclass
class SplitsDataset:
    train: BaseExperimentDataset
    val: BaseExperimentDataset
    test: BaseExperimentDataset

    @property
    def datasets(self):
        return {'train': self.train, 'val': self.val, 'test': self.test}


@dataclass
class BaseDataProvider:
    raw_data_path: str
    prepared_data_path: str

    def __post_init__(self):
        os.makedirs(self.raw_data_path, exist_ok=True)
        os.makedirs(self.prepared_data_path, exist_ok=True)

    def load_raw_data(self) -> TransposedRawData:
        """Returns raw experiments."""
        raise NotImplementedError

    def initialize_features(
            self, features: BaseDataSchema, data: BaseRawData
    ) -> BaseDataSchema:
        """Returns initialized feature objects."""
        raise NotImplementedError

    def initialize_prepared_data(
            self, features: BaseDataSchema, data: BaseRawData
    ) -> SplitsDataset:
        raise NotImplementedError

    def initialize(self, features: BaseDataSchema) -> Tuple[BaseDataSchema, SplitsDataset]:
        raise NotImplementedError


@dataclass
class BaseTrainer:
    artifacts_path: str

    def __post_init__(self):
        os.makedirs(self.artifacts_path, exist_ok=True)

    def train_epoch(self, loader, model, objective, optimizer):
        raise NotImplementedError

    def val_epoch(self, loader, model) -> Dict[str, float]:
        raise NotImplementedError

    def train(self, data: SplitsDataset, model: nn.Module) -> nn.Module:
        raise NotImplementedError


@dataclass
class BaseExperimentRunner:
    data_provider: BaseDataProvider
    trainer: BaseTrainer
    features: BaseDataSchema

    def initialize_model(
            self, features: BaseDataSchema
    ) -> nn.Module:
        raise NotImplementedError

    def run(self, **kwargs) -> 'BaseExperimentRunner':
        raise NotImplementedError
