from dataclasses import dataclass

from torch import nn

from experiments import BaseExperimentRunner
from .data import DataSchema
from .model import TransformerModel
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
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=2, batch_first=True,
            dim_feedforward=512, dropout=0.1
        )
        transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        model = TransformerModel(
            src_item_encoder=src_item_encoder,
            tgt_item_encoder=tgt_item_encoder,
            interaction_encoder=interaction_encoder,
            transformer=transformer
        )
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
