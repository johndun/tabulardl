import numpy as np
import torch
from torch import nn

from experiments.beauty2.data import _grouped_record_ids, Dataset
from experiments.beauty2.features import (
    INTERACTION_FEATURE_SCHEMA_NO_TEXT, ITEM_FEATURE_SCHEMA_NO_TEXT,
)
from experiments.beauty2 import (
    DataSchema, DataProvider, ExperimentRunner
)
from experiments.beauty2.model import TransformerModel, Trainer
from tabularencoder import get_nested_key, TabularDataset, TabularEncoder

torch_device = 'cpu'
DATA_PATH = '/Users/jdunaven/Work/workspaces/data'
ARTIFACTS_PATH = '/Users/jdunaven/Work/workspaces/experiments'

trainer_args = {
    'n_epochs': 100,
    'batch_size': 32,
    'starting_learning_rate': 0.01,
    'val_every': 1,
    'lr_reduce_factor': 0.1,
    'lr_reduce_patience': 5,  # This gets multiplied by val_every
    'early_stopping_patience': 30
}

features = DataSchema(INTERACTION_FEATURE_SCHEMA_NO_TEXT, ITEM_FEATURE_SCHEMA_NO_TEXT)
experiment = ExperimentRunner(
    features=DataSchema(INTERACTION_FEATURE_SCHEMA_NO_TEXT, ITEM_FEATURE_SCHEMA_NO_TEXT),
    data_provider=DataProvider(
        raw_data_path=f'{DATA_PATH}/amazon-beauty-2014-5core',
        prepared_data_path=f'{DATA_PATH}/beauty/no-text',
        force=False,
        use_text_features=False
    ),
    trainer=Trainer(
        artifacts_path=f'{ARTIFACTS_PATH}/beauty/no-text',
        **trainer_args
    ),
    torch_device='cpu',
)
experiment.run()
raise Exception

features, data = experiment.data_provider.initialize(features)
data.val[8]

raise Exception
model = experiment.initialize_model(features)

loader = data.val.loader(batch_size=32)

# Val
loaders = {
    split: dataset.loader(
        batch_size=8,
        # shuffle=split != 'test',
        # drop_last=split == 'train',
        # pin_memory=False
    )
    for split, dataset in data.datasets.items()
}
for batch in loaders['val']:
    break

# for batch in loader:
#     src_embeddings, tgt_embeddings, neg_embeddings = model(**batch)
#     print(src_embeddings.shape)
#     print(tgt_embeddings.shape)
#     print(neg_embeddings.shape)
#     break
#
#
#


for batch_key, batch_val in batch.items():
    if isinstance(batch_val, dict):
        print(batch_key)
        for key, val in batch_val.items():
            print(f'{key}: {val.dtype} {val.shape}')
    else:
        print(f'{batch_key}: {batch_val.dtype} {batch_val.shape}')

#
# data = [
#     {'exp': 'text', 'HR@1': 0.18418816795599874, 'HR@5': 0.4175647274515942, 'HR@10': 0.5385234539194205, 'NDCG@5': 0.30549973831275695, 'NDCG@10': 0.34461061535402715},
#     {'exp': 'no-text', 'HR@1': 0.1694763672137012, 'HR@5': 0.39292581496221435, 'HR@10': 0.5005142422751867, 'NDCG@5': 0.2853836870170027, 'NDCG@10': 0.32016190585715665},
#     {'exp': 'minimal', 'HR@1': 0.14805705853418594, 'HR@5': 0.3241067835263605, 'HR@10': 0.42154451549434335, 'NDCG@5': 0.2395648377177189, 'NDCG@10': 0.2709701637705094},
#     {'exp': 'text-no-int', 'HR@1': 0.16473639493806735, 'HR@5': 0.38420605464383134, 'HR@10': 0.4940303179358762, 'NDCG@5': 0.2796291093229688, 'NDCG@10': 0.31510727589881915},
#     {'exp': 'minimal-no-int', 'HR@1': 0.1629477261548093, 'HR@5': 0.3838930376067612, 'HR@10': 0.4833877386754908, 'NDCG@5': 0.2782454510918085, 'NDCG@10': 0.3104140540064434},
#     {'exp': 'no-text-no-int', 'HR@1': 0.1625452756785762, 'HR@5': 0.37754326342619504, 'HR@10': 0.48379018915172384, 'NDCG@5': 0.2741977125492888, 'NDCG@10': 0.3085872130045784},
# ]
# columns = ['exp', 'HR@1', 'HR@5', 'HR@10', 'NDCG@5', 'NDCG@10']
# string = ','.join(columns) + '\n'
# for record in data:
#     string += ','.join([str(record[key]) for key in columns]) + '\n'
# print(string)
