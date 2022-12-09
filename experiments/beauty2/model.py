import json
from dataclasses import dataclass
from typing import Optional, Dict

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from experiments import BaseTrainer, SplitsDataset
from tabularencoder import device_move_nested


class TransformerModel(nn.Module):
    def __init__(
            self,
            transformer: nn.TransformerEncoder,
            src_item_encoder: nn.Module,
            tgt_item_encoder: nn.Module,
            interaction_encoder: nn.Module,
    ):
        super().__init__()
        self.src_item_encoder = src_item_encoder
        self.tgt_item_encoder = tgt_item_encoder
        self.interaction_encoder = interaction_encoder
        self.transformer = transformer

    def forward(self, src_interactions, src_items, tgt_items, neg_items, mask, **_):
        src_embd = self.src_item_encoder(**src_items)
        if self.interaction_encoder:
            src_embd += self.interaction_encoder(**src_interactions)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(src_embd.shape[1])
        self.transformer(src_embd, mask=causal_mask, src_key_padding_mask=mask)
        tgt_embd = self.tgt_item_encoder(**tgt_items)
        neg_embd = self.tgt_item_encoder(**neg_items)
        return src_embd, tgt_embd, neg_embd

    def save_to(self, path):
        torch.save(self, f'{path}/model.torch')

    def load_from(self, path):
        return torch.load(f'{path}/model.torch')


@dataclass
class Trainer(BaseTrainer):
    margin: float = 0.5
    n_epochs: int = 100
    batch_size: int = 32
    starting_learning_rate: float = 0.1
    lr_reduce_factor: float = 0.1
    lr_reduce_patience: int = 1
    val_every: int = 1
    early_stopping_patience: int = 1
    torch_device: str = 'cpu'

    def train_epoch(self, loader, model, optimizer, objective):
        model.train()
        torch_device = torch.device(self.torch_device)
        total_loss = 0
        total_samples = 0
        for batch in tqdm(loader):
            optimizer.zero_grad()
            src_embd, tgt_embd, neg_embd = model(**device_move_nested(batch, torch_device))
            src_embd = src_embd.view(-1, src_embd.shape[-1])
            tgt_embd = tgt_embd.view(-1, src_embd.shape[-1])
            neg_embd = neg_embd.view(-1, src_embd.shape[-1])
            batch_size = src_embd.shape[0]
            loss = objective(src_embd, tgt_embd, neg_embd)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_samples += batch_size
        return total_loss / total_samples

    def val_epoch(self, loader, model) -> Dict[str, float]:
        model.eval()
        torch_device = torch.device(self.torch_device)
        total = 0
        ks = [1, 5, 10]
        hits = [0, 0, 0]
        ndcgs = [0, 0, 0]
        for batch in tqdm(loader):
            with torch.no_grad():
                src_embd, tgt_embd, neg_embd = model(**device_move_nested(batch, torch_device))
                src_embd = src_embd[:, -1:, :]
                tgt_embd = torch.concat((tgt_embd, neg_embd), dim=1)
                scores = torch.bmm(src_embd, tgt_embd.transpose(1, 2))[:, 0, :]
                total += scores.shape[0]
                hit_cnts, ndcg = _score_batch(scores, ks, torch_device)
                for idx in range(len(ks)):
                    hits[idx] += hit_cnts[idx]
                    ndcgs[idx] += ndcg[idx]
        metrics = {f'HR@{k}': hit_cnts / total for k, hit_cnts in zip(ks, hits)}
        metrics['NDCG@5'] = ndcgs[1] / total
        metrics['NDCG@10'] = ndcgs[2] / total
        print(scores[0, :8])
        return metrics

    def train(self, data: SplitsDataset, model: nn.Module) -> nn.Module:
        torch_device = torch.device(self.torch_device)
        model = model.to(torch_device)
        objective = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y),
            margin=self.margin,
            reduction='sum'
        )
        objective = objective.to(torch_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.starting_learning_rate)
        loaders = {
            split: dataset.loader(
                batch_size=self.batch_size,
                shuffle=split != 'test',
                drop_last=split == 'train',
                pin_memory=self.torch_device != 'cpu'
            )
            for split, dataset in data.datasets.items()
        }
        scheduler = ReduceLROnPlateau(
            optimizer,
            factor=self.lr_reduce_factor,
            patience=self.lr_reduce_patience,
            threshold=0.,
            verbose=True
        )
        metrics = []
        best_epoch = 0
        best_loss = 1e10
        for epoch in range(self.n_epochs):
            train_loss = self.train_epoch(
                loader=loaders['train'],
                model=model, optimizer=optimizer, objective=objective
            )
            if (epoch + 1) % self.val_every and epoch < self.n_epochs - 1:
                continue
            epoch_metrics = self.val_epoch(loader=loaders['val'], model=model)
            epoch_metrics['train_loss'] = train_loss
            epoch_metrics['epoch'] = epoch + 1
            print(epoch_metrics)
            val_loss = -1 * epoch_metrics['HR@10']
            if val_loss < best_loss:
                best_epoch = epoch
                best_loss = val_loss
                model.save_to(self.artifacts_path)
            if (epoch - best_epoch) >= self.early_stopping_patience:
                print(f'Stopping. No improvement in {epoch - best_epoch} epochs')
                break
            scheduler.step(val_loss)
            metrics.append(epoch_metrics)
        model = model.load_from(self.artifacts_path).to(torch_device)
        test_metrics = self.val_epoch(loader=loaders['test'], model=model)
        print('Test set results:')
        print(test_metrics)
        json.dump(test_metrics, open(f'{self.artifacts_path}/test_set_results.json', 'w'))
        return model


def _score_batch(scores, ks, torch_device):
    ks = ks or [1, 5, 10]
    hits = [0] * len(ks)
    ndcgs = [0] * len(ks)
    score_sort_indices = torch.argsort(scores, dim=-1, descending=True)
    denoms = torch.log2(
        torch.LongTensor([range(scores.shape[-1])]).expand(scores.shape) + 2
    ).to(torch_device)
    for idx, k in enumerate(ks):
        hits[idx] = (score_sort_indices[:, :k] == 0).float().sum().item()
        if k == 1:
            continue
        ndcgs[idx] = ((score_sort_indices[:, :k] == 0).float() / denoms[:, :k]).sum().item()
    return hits, ndcgs
