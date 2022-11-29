from dataclasses import dataclass
import csv

from tabulardl import *


@dataclass
class Args(RuntimeArgs):
    data_basepath: str = '/Users/jdunaven/Work/workspaces/data/titanic'
    artifacts_basepath: str = '/Users/jdunaven/Work/workspaces/experiments/titanic'


args = Args().parse_args()
train_data, data_schema = TitanicDataset(args.data_basepath).get_training_data()
_ = [feat.fit_data_transformer(train_data[key]) for key, feat in data_schema.items()]
dataset = TabularDataset({
    key: feat.transform_raw_data(train_data[key]) for key, feat in data_schema.items()
})

print(*[f'{k}: {v[:5]}' for k, v in train_data.items()],sep='\n')
print(*[(k, v) for k, v in data_schema.items()], sep='\n')


print(*[f'{key}:\t{val.shape}\t{val.dtype}' for key, val in dataset[:2].items()], sep='\n')




# def main(args: RuntimeArgs):
#     print(args)
#     columns, data = load_data()


# if __name__ == '__main__':
#     args = Args().parse_args()
#     main(args)
