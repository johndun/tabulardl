import numpy as np

from tabulardl import (
    TabularDataset, NumericFeature, CategoricalFeature, CategoricalArrayFeature
)

x1 = np.random.normal(1., 1., size=10).tolist()
x2 = np.random.normal(1., 1., size=(10, 1)).tolist()
x3 = np.random.normal(1., 1., size=(10, 2)).tolist()

feature = NumericFeature()


lower, upper = np.percentile(x3, q=[0, 75], axis=0)
np.clip(x3, a_min=lower, a_max=upper)


n_samples = 100
data = {
    'y': np.random.normal(0., 1., size=n_samples).tolist(),
    'x0': np.random.choice(['a', 'b'], size=n_samples).tolist(),
    'x1': np.random.normal(0., 1., size=n_samples).tolist(),
    'x2': np.random.choice(['a', 'b'], size=(n_samples, 3)).tolist(),
}

# Define schema
SAMPLE_SCHEMA = {
    'y': NumericFeature(),
    'x0': CategoricalFeature(),
    'x1': NumericFeature(),
    'x2': CategoricalArrayFeature(max_len=3),
}
# Initialize states
_ = [feat.fit_data_transformer(data[key]) for key, feat in SAMPLE_SCHEMA.items()]
# Transform raw data
dataset = TabularDataset({
    key: feat.transform_raw_data(data[key]) for key, feat in SAMPLE_SCHEMA.items()
})
print(*[f'{key}:\t{val.shape}\t{val.dtype}' for key, val in dataset[:2].items()], sep='\n')


data = {
    'action': np.random.choice(['a', 'b'], size=n_samples).tolist(),
    'reward': np.random.normal(0., 1., size=n_samples).tolist(),
    'state': {
        'state_feature0': np.random.normal(0., 1., size=n_samples).tolist(),
        'state_feature1': np.random.choice(['a', 'b'], size=(n_samples, 3)).tolist(),
    },
    'next_state': {
        'state_feature0': np.random.normal(0., 1., size=n_samples).tolist(),
        'state_feature1': np.random.choice(['a', 'b'], size=(n_samples, 3)).tolist(),
    }
}

STATE_SCHEMA = {
    'state_feature0': NumericFeature(center=True, scale=True, clip_percentiles=[None, 0.99]),
    'state_feature1': CategoricalArrayFeature(max_len=3),
}
SAMPLE_SCHEMA = {
    'action': CategoricalFeature(),
    'reward': NumericFeature(),
}

_ = [feature.fit_data_transformer(data['state'][key]) for key, feature in STATE_SCHEMA.items()]
_ = [feature.fit_data_transformer(data[key]) for key, feature in SAMPLE_SCHEMA.items()]

dataset = TabularDataset({
    'action': SAMPLE_SCHEMA['action'].transform_raw_data(data['action']),
    'reward': SAMPLE_SCHEMA['reward'].transform_raw_data(data['reward']),
    'state': TabularDataset({
        key: STATE_SCHEMA[key].transform_raw_data(datum)
        for key, datum in data['state'].items()
    }),
    'next_state': TabularDataset({
        key: STATE_SCHEMA[key].transform_raw_data(datum)
        for key, datum in data['next_state'].items()
    })
})

print(*[f'{key}:\t{val}' for key, val in dataset[:2].items()], sep='\n')
