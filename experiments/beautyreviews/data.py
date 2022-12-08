import ast
import gzip
import json
import os
import pickle
import urllib
from dataclasses import dataclass
from functools import partial
from itertools import chain
from typing import List, Dict, Optional, Set

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from experiments import (
    BaseRawData, BaseDataSchema, BaseExperimentDataset, TransposedRawData,
    SplitsDataset, BaseDataProvider
)
from jutils import Timer
from tabularencoder import (
    Feature, TabularDataset, transpose_json_data, infer_features,
    get_nested_key
)


@dataclass
class RawData(BaseRawData):
    interaction: TransposedRawData
    item: TransposedRawData


@dataclass
class DataSchema(BaseDataSchema):
    interaction: Optional[Dict[str, Feature]]
    item: Dict[str, Feature]


@dataclass
class Dataset(BaseExperimentDataset):
    interaction: TabularDataset
    item: TabularDataset
    user_sample_id_map: List[List[int]]
    interaction_item_map: List[List[int]]
    split: str = 'train'
    item_ids: Set = None
    size: int = None

    def __post_init__(self):
        self.item_ids = set(range(len(self.item)))
        self.size = self.size or 1 if self.split == 'train' else 99

    def __getitem__(self, idx):
        if self.split == 'train':
            sample_ids = self.user_sample_id_map[idx][:-2]
        elif self.split == 'val':
            sample_ids = self.user_sample_id_map[idx][-3:-1]
        else:
            sample_ids = self.user_sample_id_map[idx][-2:]

        all_samples = set(self.user_sample_id_map[idx])
        if self.size == 1:
            negative_sample_ids = np.random.choice(len(self.item_ids), size=1)
            while negative_sample_ids[0] in all_samples:
                negative_sample_ids = np.random.choice(len(self.item_ids), size=1)
        else:
            negative_sample_ids = np.random.choice(
                list(self.item_ids - all_samples),
                size=self.size,
                replace=False
            )
        if len(sample_ids) > 2:
            random_indices = sorted(np.random.choice(len(sample_ids), size=2, replace=False))
            sample_ids = [sample_ids[idx] for idx in random_indices]

        interaction = self.interaction[sample_ids[0]]
        item = self.item[self.interaction_item_map[sample_ids[0]]]
        target = self.interaction_item_map[sample_ids[1]]
        return {
            'interaction': interaction,
            'item': item,
            'target': self.item[target],
            'neg_targets': self.item[negative_sample_ids]
        }

    def __len__(self):
        return len(self.user_sample_id_map)

    @staticmethod
    def load_from(path: str) -> SplitsDataset:
        interaction_item_map, user_sample_id_map = pickle.load(open(
            f'{path}/train_artifacts.pickle', 'rb'
        ))
        interaction_data = TabularDataset.load_from(f'{path}/interaction_data.torch')
        item_data = TabularDataset.load_from(f'{path}/item_data.torch')
        get_data = partial(
            Dataset,
            interaction=interaction_data,
            item=item_data,
            user_sample_id_map=user_sample_id_map,
            interaction_item_map=interaction_item_map,
        )
        return SplitsDataset(
            **{split: get_data(split=split) for split in ('train', 'val', 'test')}
        )

    def save_to(self, path: str):
        self.interaction.save_to(f'{path}/interaction_data.torch')
        self.item.save_to(f'{path}/item_data.torch')
        pickle.dump(
            (self.interaction_item_map, self.user_sample_id_map),
            open(f'{path}/train_artifacts.pickle', 'wb')
        )

    def loader(self, **kwargs):
        return DataLoader(self, **kwargs)


def _grouped_record_ids(users, timestamps):
    """Provides interaction indices for each user."""
    user_record_id_map = {}
    for idx, (user, timestamp) in enumerate(zip(users, timestamps)):
        user_ids = user_record_id_map.get(user, [])
        user_ids.append((idx, timestamp))
        user_record_id_map[user] = user_ids
    grouped_record_ids = [
        [x[0] for x in sorted(ids, key=lambda x: x[1])]
        for ids in user_record_id_map.values()
    ]
    return grouped_record_ids


def encode_string_pretrained_lm(data, tokenizer, text_encoder, torch_device):
    tokenized_strings = tokenizer(data, padding=True, truncation=True, return_tensors="pt")
    dataset = TabularDataset({k: v for k, v in tokenized_strings.items() if k != 'token_type_ids'})
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )
    embeddings = []
    for batch in tqdm(loader):
        with torch.no_grad():
            batch_embeddings = text_encoder(
                input_ids=batch['input_ids'].to(torch_device),
                attention_mask=batch['attention_mask'].to(torch_device),
            ).last_hidden_state[:, 0, :]
        embeddings.append(batch_embeddings.to('cpu'))
    embeddings = torch.concat(embeddings, dim=0).numpy()
    return embeddings


@dataclass
class DataProvider(BaseDataProvider):
    use_text_features: bool = False
    force: bool = True
    torch_device: str = 'cpu'

    def load_raw_data(self) -> RawData:
        timer = Timer()
        # Download base data if needed
        data_files = (
            (
                f'{self.raw_data_path}/meta_Beauty.json.gz',
                'http://snap.stanford.edu/data/amazon/productGraph/'
                'categoryFiles/meta_Beauty.json.gz'
            ),
            (
                f'{self.raw_data_path}/reviews_Beauty_5.json.gz',
                'http://snap.stanford.edu/data/amazon/productGraph/'
                'categoryFiles/reviews_Beauty_5.json.gz'
            )
        )
        for path, url in data_files:
            if not os.path.exists(path):
                print(f'Downloading: {url}')
                os.makedirs(self.raw_data_path, exist_ok=True)
                urllib.request.urlretrieve(url, path)
                print(f'  Finished in {timer():,.2f} seconds')

        # Import data
        print('Initializing interaction data')
        interaction_data = []
        with gzip.open(f'{self.raw_data_path}/reviews_Beauty_5.json.gz', 'r') as jsonfile:
            for idx, line in enumerate(jsonfile):
                interaction_data.append(json.loads(line))
        print(f'  Finished in {timer():,.2f} seconds')

        print('Initializing item data')
        with gzip.open(f'{self.raw_data_path}/meta_Beauty.json.gz', 'r') as jsonfile:
            item_data = [ast.literal_eval(line.decode()) for line in jsonfile]

        interaction_items = set([sample['asin'] for sample in interaction_data])
        item_data = [sample for sample in item_data if sample['asin'] in interaction_items]
        print(f'  Finished in {timer():,.2f} seconds')

        """Prepocess raw data."""
        for item in item_data:
            item['categories'] = item['categories'][0]
            if 'salesRank' in item and 'Beauty' in item['salesRank']:
                item['salesRank']['Beauty'] = -1 * item['salesRank']['Beauty']
            item['title'] = item.get('title', '') or ''
            item['description'] = item.get('description', '') or ''

        for interaction in interaction_data:
            interaction['review_month'], interaction['review_day'], interaction['review_year'] = (
                [int(x.strip(',')) for x in interaction['reviewTime'].split()]
                if 'reviewTime' in interaction and interaction['reviewTime'] else
                [None, None, None]
            )

        interaction_data, interaction_feature_meta = transpose_json_data(interaction_data)
        item_data, item_feature_meta = transpose_json_data(item_data)

        if self.use_text_features:
            torch_device = torch.device(self.torch_device)
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            text_encoder = AutoModel.from_pretrained("distilbert-base-uncased").to(torch_device)
            for field in ('summary', 'reviewText'):
                print(field)
                strings = interaction_data[field]
                interaction_data[field] = encode_string_pretrained_lm(
                    strings, tokenizer, text_encoder, torch_device
                )
            for field in ('title', 'description'):
                print(field)
                strings = item_data[field]
                item_data[field] = encode_string_pretrained_lm(
                    strings, tokenizer, text_encoder, torch_device
                )

        for dataset, data in zip(
                ('interaction', 'item'), (interaction_feature_meta, item_feature_meta)
        ):
            print(f'Inferred features for: {dataset}')
            infer_features(data)
            print(' ')

        return RawData(interaction_data, item_data)

    def initialize_features(self, features: DataSchema, data: RawData) -> DataSchema:
        users = data.interaction['reviewerID']
        timestamps = data.interaction['unixReviewTime']
        grouped_record_ids = _grouped_record_ids(users, timestamps)
        # Remove last 2 interactions from train
        train_ids = [ids[:-2] for ids in grouped_record_ids]
        for label, features_, data_ in (
                ('interaction', features.interaction, data.interaction),
                ('item', features.item, data.item),
        ):
            for feature in features_.values():
                feature_data = get_nested_key(data_, feature.key)
                feature.fit_data_transformer(
                    [feature_data[idx] for idx in chain(*train_ids)]
                    if label == 'interaction' else
                    feature_data
                )
        return features

    def initialize_prepared_data(self, features: DataSchema, data: RawData) -> SplitsDataset:
        users = data.interaction['reviewerID']
        timestamps = data.interaction['unixReviewTime']
        grouped_record_ids = _grouped_record_ids(users, timestamps)
        interaction_items = data.interaction['asin']
        item_data_lookup = {item_id: idx for idx, item_id in enumerate(data.item['asin'])}
        interaction_item_map = [item_data_lookup[item] for item in interaction_items]

        transformed_data = {'interaction': {}, 'item': {}}
        for key, features_, data_ in (
                ('interaction', features.interaction, data.interaction),
                ('item', features.item, data.item),
        ):
            for feature in features_.values():
                feature_data = get_nested_key(data_, feature.key)
                transformed_data[key][feature.id] = feature.transform_raw_data(feature_data)
        datasets = {key: TabularDataset(val) for key, val in transformed_data.items()}
        return SplitsDataset(**{
            split: Dataset(
                **datasets,
                split=split,
                interaction_item_map=interaction_item_map,
                user_sample_id_map=grouped_record_ids
            )
            for split in ('train', 'val', 'test')
        })

    def initialize(self, features):
        artifact_paths = [
            f'{self.prepared_data_path}/data_schema.pickle',
            f'{self.prepared_data_path}/train_artifacts.pickle',
            f'{self.prepared_data_path}/interaction_data.torch',
            f'{self.prepared_data_path}/item_data.torch'
        ]
        if all(os.path.exists(x) for x in artifact_paths) and not self.force:
            loaded_features = DataSchema.load_from(self.prepared_data_path)
            data = Dataset.load_from(self.prepared_data_path)
            features.interaction = {
                key: loaded_features.interaction[key] for key in features.interaction.keys()
            }
            data.train.interaction.data = {
                key: data.train.interaction.data[key] for key in features.interaction.keys()
            }
            features.item = {
                key: loaded_features.item[key] for key in features.item.keys()
            }
            data.train.item.data = {
                key: data.train.item.data[key] for key in features.item.keys()
            }
        else:
            raw_data = self.load_raw_data()
            features = self.initialize_features(features, raw_data)
            features.save_to(self.prepared_data_path)
            data = self.initialize_prepared_data(features, raw_data)
            data.train.save_to(self.prepared_data_path)
        return features, data
