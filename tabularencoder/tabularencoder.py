from typing import List, Optional

import torch
from torch import nn

from .features import *


class TabularEncoder(nn.Module):
    def __init__(
            self,
            embedding_size: int = 128,
            numeric: Optional[List[NumericFeature]] = None,
            categorical: Optional[List[CategoricalFeature]] = None,
            categorical_array: Optional[List[CategoricalArrayFeature]] = None,
            concat: bool = False
    ):
        self.embedding_size = embedding_size
        self.concat = concat
        super().__init__()
        self.embedders = nn.ModuleDict({
            feature.id.replace('.', '_'): nn.Embedding(
                num_embeddings=feature.dictionary_size,
                embedding_dim=self.embedding_size
            )
            for feature in categorical or []
        })
        self.array_embedders = nn.ModuleDict({
            feature.id.replace('.', '_'): nn.EmbeddingBag(
                num_embeddings=feature.dictionary_size,
                embedding_dim=self.embedding_size,
                padding_idx=0,
                mode='mean'
            )
            for feature in categorical_array or []
        })
        self.numeric_projections = nn.ModuleDict({
            feature.id.replace('.', '_'): nn.Linear(
                in_features=1
                if isinstance(feature.mean, float) or not feature.mean.shape else
                feature.mean.shape[0],
                out_features=self.embedding_size
            )
            for feature in numeric or []
        })

    def forward(self, **features):
        hidden = []
        for modules in (self.numeric_projections, self.embedders, self.array_embedders):
            for feat_name, layer in modules.items():
                inp = features[feat_name]
                inp_shape = inp.shape
                hidden.append(
                    layer(inp)
                    if len(inp_shape) <= 2 else
                    layer(inp.view(-1, inp_shape[-1])).view(inp_shape[0], inp_shape[1], -1)
                )
        if self.concat:
            return torch.concat(hidden, dim=-1)
        return torch.stack(hidden, dim=-1).sum(-1)
