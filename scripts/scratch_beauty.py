import ast
import gzip
import json
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain

import numpy as np
import torch
from tqdm import tqdm

from transformers import AutoTokenizer
from transformers import AutoModel

from tabulardl import *


@dataclass
class Args(RuntimeArgs):
    data_basepath: str = '/Users/jdunaven/Work/workspaces/data/amazon-beauty-2014-5core'
    artifacts_basepath: str = '/Users/jdunaven/Work/workspaces/experiments/amazon-beauty-2014-5core'


args = Args().parse_args()
print(args)

with gzip.open(f'{args.data_basepath}/meta_Beauty.json.gz', 'r') as jsonfile:
    item_data = [ast.literal_eval(line.decode()) for line in jsonfile]

with gzip.open(f'{args.data_basepath}/reviews_Beauty_5.json.gz', 'r') as jsonfile:
    interaction_data = [json.loads(line) for line in jsonfile]

interaction_items = set([sample['asin'] for sample in interaction_data])
item_data = [sample for sample in item_data if sample['asin'] in interaction_items]

for idx in range(500, 505):
    print(*[(key, val) for key, val in item_data[idx].items()], sep='\n')

for idx in range(500, 505):
    print(*[(key, val) for key, val in interaction_data[idx].items()], sep='\n')


for field in ('also_bought', 'also_viewed', 'bought_together'):
    lens = []
    for item in item_data:
        lens.append(len(item.get('related', {}).get(field, [])))
    print(f'{field}: max: {np.max(lens)}; mean: {np.mean(lens):,.3f}')

for field in ('title',):
    strings = [sample.get(field, '') for sample in item_data]

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text_encoder = AutoModel.from_pretrained("distilbert-base-uncased")

embeddings = []
for batch in tqdm(chunked_iterator(strings, 8)):
    batch_tokens = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
    batch_embeddings = text_encoder(
        input_ids=batch_tokens['input_ids'], attention_mask=batch_tokens['attention_mask']
    ).last_hidden_state[:, 0, :]
    embeddings.append(batch_embeddings)

embeddings = torch.concat(embeddings, dim=0)


item_features = {
    'asin': CategoricalFeature(max_vocab_size=100000),
    'related': {
        'also_bought': CategoricalArrayFeature(max_vocab_size=100000, max_len=100),
        'also_viewed': CategoricalArrayFeature(max_vocab_size=100000, max_len=60),
        'bought_together': CategoricalArrayFeature(max_vocab_size=100000, max_len=4)
    },
    # 'reviewerID': CategoricalFeature(),
    # 'vote': NumericFeature(missing_value=0.),
    # 'overall': NumericFeature(),
    # 'verified': NumericFeature(missing_value=0.),
}



# filtered_users = set([id for id, cnt in user_counts.items() if cnt >= 5])
# filtered_items = set([id for id, cnt in item_counts.items() if cnt >= 5])
# filtered_data = []
# for sample in data:
#     if (
#             sample['reviewerID'] in filtered_users
#             and sample['asin'] in filtered_items
#     ):
#         filtered_data.append(sample)
