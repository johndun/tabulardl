from experiments.beautyreviews import (
    INTERACTION_FEATURE_SCHEMA_MINIMAL, ITEM_FEATURE_SCHEMA_MINIMAL,
    INTERACTION_FEATURE_SCHEMA_NO_TEXT, ITEM_FEATURE_SCHEMA_NO_TEXT,
    DataSchema, ExperimentRunner, DataProvider, Trainer,
)

DATA_PATH = '/Users/jdunaven/Work/workspaces/data'
ARTIFACTS_PATH = '/Users/jdunaven/Work/workspaces/experiments'

experiment1 = ExperimentRunner(
    features=DataSchema(INTERACTION_FEATURE_SCHEMA_NO_TEXT, ITEM_FEATURE_SCHEMA_NO_TEXT),
    data_provider=DataProvider(
        raw_data_path=f'{DATA_PATH}/amazon-beauty-2014-5core',
        prepared_data_path=f'{DATA_PATH}/beauty/no-text',
        force=False
    ),
    trainer=Trainer(
        artifacts_path=f'{ARTIFACTS_PATH}/beauty/no-text',
        n_epochs=1,
        early_stopping_patience=10
    ),
    # torch_device='cuda:0'
).run()

# experiment2 = ExperimentRunner(
#     features=DataSchema(INTERACTION_FEATURE_SCHEMA_MINIMAL, ITEM_FEATURE_SCHEMA_MINIMAL),
#     data_provider=DataProvider(
#         raw_data_path=f'{DATA_PATH}/amazon-beauty-2014-5core',
#         prepared_data_path=f'{DATA_PATH}/beauty/no-text',
#         force=False
#     ),
#     trainer=Trainer(
#         artifacts_path=f'{ARTIFACTS_PATH}/beauty/minimal',
#         n_epochs=0
#     ),
# ).run()
#
# experiment3 = ExperimentRunner(
#     features=DataSchema({}, ITEM_FEATURE_SCHEMA_MINIMAL),
#     data_provider=DataProvider(
#         raw_data_path=f'{DATA_PATH}/amazon-beauty-2014-5core',
#         prepared_data_path=f'{DATA_PATH}/beauty/no-text',
#         force=False
#     ),
#     trainer=Trainer(
#         artifacts_path=f'{ARTIFACTS_PATH}/beauty/minimal-no-interaction',
#         n_epochs=0
#     ),
# ).run()
