"""
https://huggingface.co/docs/transformers/autoclass_tutorial
https://huggingface.co/docs/transformers/preprocessing
https://towardsdatascience.com/feature-extraction-with-bert-for-text-classification-533dde44dc2f
"""
import torch
from torch import nn

# transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
# src = torch.rand((10, 32, 512))
# tgt = torch.rand((20, 32, 512))
# out = transformer_model(src, tgt)

encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
src = torch.rand(10, 32, 512)
msk = nn.Transformer.generate_square_subsequent_mask(10, device='cpu')
out = transformer_encoder(src, msk)
