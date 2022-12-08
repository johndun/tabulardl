import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, ffn, sz):
        super().__init__()
        self.ffn = ffn
        self.norm = nn.LayerNorm(sz)

    def forward(self, x):
        return self.ffn(self.norm(x)) + x


class Model(nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.net = nn.Sequential(*[
            ResBlock(self._get_ff(self.hidden_size), self.hidden_size)
            for _ in range(self.n_blocks)
        ])

    def _get_ff(self, sz):
        return nn.Sequential(
            nn.Linear(sz, sz * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(sz * 2, sz),
            nn.Dropout(self.dropout)
        )

    def forward(self, x):
        hidden = self.net(x)
        return hidden


model = Model()
x = torch.rand((4, 8))
output = model(x)
