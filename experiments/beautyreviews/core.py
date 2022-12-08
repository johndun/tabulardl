from dataclasses import dataclass

import torch
from torch import nn

from experiments import BaseExperimentRunner
from .data import DataSchema
from .model import ShallowModel
from tabularencoder import TabularEncoder


@dataclass
class ExperimentRunner(BaseExperimentRunner):
    torch_device: str = 'cpu'

    def initialize_model(self, features: DataSchema) -> nn.Module:
        schemas = {
            'interaction': DataSchema.group_features_by_type(features.interaction),
            'item': DataSchema.group_features_by_type(features.item)
        }
        src_item_encoder, tgt_item_encoder = (
            TabularEncoder(
                **schemas['item']
            )
            for _ in range(2)
        )
        interaction_encoder = (
            TabularEncoder(
                **schemas['interaction']
            )
            if len(features.interaction) else
            None
        )
        model = ShallowModel(src_item_encoder, tgt_item_encoder, interaction_encoder)
        return model

    def run(self):
        self.data_provider.torch_device = self.torch_device
        self.trainer.torch_device = self.torch_device
        features, data = self.data_provider.initialize(self.features)
        features.save_to(self.trainer.artifacts_path)
        model = self.initialize_model(features)
        model.save_to(self.trainer.artifacts_path)
        print(features)
        print(model)
        _ = self.trainer.train(data=data, model=model)
        return self
