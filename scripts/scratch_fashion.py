import json
from dataclasses import dataclass
from itertools import chain

from tabulardl import *


@dataclass
class Args(RuntimeArgs):
    data_basepath: str = '/Users/jdunaven/Work/workspaces/data/amazon-fashion'
    artifacts_basepath: str = '/Users/jdunaven/Work/workspaces/experiments/amazon-fashion'


def flip(data):
    keys = set(chain(*[(x or {}).keys() for x in data]))
    flipped = {key: [(x or {}).get(key, None) for x in data] for key in keys}
    na_count = {key: len([x for x in val if x is not None]) for key, val in flipped.items()}
    print('Nonempty records:')
    print(*[f'{k}: {v}' for k, v in na_count.items()], sep='\n')
    examples = {key: [x for x in val if x is not None][:5] for key, val in flipped.items()}
    print('\nExamples:')
    print(*[f'{k}: {v}' for k, v in examples.items()], sep='\n')
    print(' ')
    return flipped


args = Args().parse_args()
print(args)

data = []
with open(f'{args.data_basepath}/meta_AMAZON_FASHION.json', 'r') as jsonfile:
    for idx, line in enumerate(jsonfile):
        print(line)
        if idx == 20:
            raise Exception
        data.append(json.loads(line))



data = []
with open(f'{args.data_basepath}/AMAZON_FASHION_5.json', 'r') as jsonfile:
    for line in jsonfile:
        data.append(json.loads(line))

train_data = flip(data)
nested_keys = ['style']
for nst_key in nested_keys:
    print(f'Nested key: {nst_key}')
    train_data[nst_key] = flip(train_data[nst_key])

train_data['verified'] = [float(x) if x is not None else None for x in train_data['verified']]
train_data['vote'] = [float(x) if x is not None else None for x in train_data['vote']]

top_level_features = {
    'asin': CategoricalFeature(max_vocab_size=100000),
    'reviewerID': CategoricalFeature(),
    'vote': NumericFeature(missing_value=0.),
    'overall': NumericFeature(),
    'verified': NumericFeature(missing_value=0.),
}
nested_features = {
    'style': {
        'Size:': CategoricalFeature(max_vocab_size=10000),
        'Color:': CategoricalFeature(max_vocab_size=10000),
    }
}

_ = [feat.fit_data_transformer(train_data[key]) for key, feat in top_level_features.items()]
for nst_key, features in nested_features.items():
    _ = [feat.fit_data_transformer(train_data[nst_key][key]) for key, feat in features.items()]

dataset = TabularDataset({
    key: feat.transform_raw_data(train_data[key]) for key, feat in top_level_features.items()
})
for nst_key, features in nested_features.items():
    dataset.data[nst_key] = TabularDataset({
        key: feat.transform_raw_data(train_data[nst_key][key]) for key, feat in features.items()
    })
