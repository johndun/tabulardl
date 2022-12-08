# !pip3 install transformers

# Sagemaker notebook specific stuff
import sys
if '/home/ec2-user/SageMaker/tabulardl' not in sys.path:
    sys.path.append('/home/ec2-user/SageMaker/tabulardl')

torch_device = 'cuda:0'
DATA_PATH = '/home/ec2-user/SageMaker/data'
ARTIFACTS_PATH = '/home/ec2-user/SageMaker/experiments'

from experiments.beautyreviews import (
    INTERACTION_FEATURE_SCHEMA_MINIMAL, ITEM_FEATURE_SCHEMA_MINIMAL,
    INTERACTION_FEATURE_SCHEMA_NO_TEXT, ITEM_FEATURE_SCHEMA_NO_TEXT,
    INTERACTION_FEATURE_SCHEMA_TEXT, ITEM_FEATURE_SCHEMA_TEXT,
    DataSchema, ExperimentRunner, DataProvider, Trainer,
)

trainer_args = {
    'n_epochs': 300,
    'batch_size': 512,
    'starting_learning_rate': 0.01,
    'val_every': 10,
    'lr_reduce_factor': 0.1,
    'lr_reduce_patience': 1,  # This gets multiplied by val_every
    'early_stopping_patience': 30
}

# # Full interaction and item feature sets
# experiment1 = ExperimentRunner(
#     features=DataSchema(INTERACTION_FEATURE_SCHEMA_TEXT, ITEM_FEATURE_SCHEMA_TEXT),
#     data_provider=DataProvider(
#         raw_data_path=f'{DATA_PATH}/amazon-beauty-2014-5core',
#         prepared_data_path=f'{DATA_PATH}/beauty/text',
#         force=False,
#         use_text_features=True
#     ),
#     trainer=Trainer(
#         artifacts_path=f'{ARTIFACTS_PATH}/beauty/text',
#         **trainer_args
#     ),
#     torch_device='cuda:0',
# ).run()

# Interaction and item features; no text
experiment2 = ExperimentRunner(
    features=DataSchema(INTERACTION_FEATURE_SCHEMA_NO_TEXT, ITEM_FEATURE_SCHEMA_NO_TEXT),
    data_provider=DataProvider(
        raw_data_path=f'{DATA_PATH}/amazon-beauty-2014-5core',
        prepared_data_path=f'{DATA_PATH}/beauty/text',
        force=False,
        use_text_features=False
    ),
    trainer=Trainer(
        artifacts_path=f'{ARTIFACTS_PATH}/beauty/no-text',
        **trainer_args
    ),
    torch_device='cuda:0',
).run()

# User and item embeddings
experiment3 = ExperimentRunner(
    features=DataSchema(INTERACTION_FEATURE_SCHEMA_MINIMAL, ITEM_FEATURE_SCHEMA_MINIMAL),
    data_provider=DataProvider(
        raw_data_path=f'{DATA_PATH}/amazon-beauty-2014-5core',
        prepared_data_path=f'{DATA_PATH}/beauty/text',
        force=False,
        use_text_features=False
    ),
    trainer=Trainer(
        artifacts_path=f'{ARTIFACTS_PATH}/beauty/minimal',
        **trainer_args
    ),
    torch_device='cuda:0',
).run()

# No interaction features; Full item features
experiment4 = ExperimentRunner(
    features=DataSchema({}, ITEM_FEATURE_SCHEMA_TEXT),
    data_provider=DataProvider(
        raw_data_path=f'{DATA_PATH}/amazon-beauty-2014-5core',
        prepared_data_path=f'{DATA_PATH}/beauty/text',
        force=False,
        use_text_features=True
    ),
    trainer=Trainer(
        artifacts_path=f'{ARTIFACTS_PATH}/beauty/text-no-int',
        **trainer_args
    ),
    torch_device='cuda:0',
).run()

# No interaction features; No text item features
experiment5 = ExperimentRunner(
    features=DataSchema({}, ITEM_FEATURE_SCHEMA_NO_TEXT),
    data_provider=DataProvider(
        raw_data_path=f'{DATA_PATH}/amazon-beauty-2014-5core',
        prepared_data_path=f'{DATA_PATH}/beauty/text',
        force=False,
        use_text_features=False
    ),
    trainer=Trainer(
        artifacts_path=f'{ARTIFACTS_PATH}/beauty/no-text-no-int',
        **trainer_args
    ),
    torch_device='cuda:0',
).run()

# Only item embeddings
experiment6 = ExperimentRunner(
    features=DataSchema({}, ITEM_FEATURE_SCHEMA_NO_TEXT),
    data_provider=DataProvider(
        raw_data_path=f'{DATA_PATH}/amazon-beauty-2014-5core',
        prepared_data_path=f'{DATA_PATH}/beauty/text',
        force=False,
        use_text_features=False
    ),
    trainer=Trainer(
        artifacts_path=f'{ARTIFACTS_PATH}/beauty/minimal-no-int',
        **trainer_args
    ),
    torch_device='cuda:0',
).run()
