import csv
from dataclasses import dataclass

from ..tabulardataset import TabularDataset
from ..utils import convert_raw_data
from ..features import CategoricalFeature, CategoricalArrayFeature, NumericFeature


@dataclass
class TitanicDataset:
    data_basepath: str

    def get_schema(self):
        return {
            'Survived': NumericFeature(),
            'Pclass': CategoricalFeature(),
            'Name': CategoricalArrayFeature(max_len=8, max_vocab_size=20),
            'Sex': CategoricalFeature(),
            'Age': NumericFeature(),
            'SibSp': NumericFeature(),
            'Parch': CategoricalFeature(max_vocab_size=20),
            'Ticket': CategoricalFeature(max_vocab_size=20),
            'Fare': NumericFeature(),
            'Cabin': CategoricalFeature(max_vocab_size=20),
            'Embarked': CategoricalFeature(max_vocab_size=20),
        }

    def get_training_data(self):
        with open(f'{self.data_basepath}/train.csv', 'r') as csvfile:
            data = list(csv.DictReader(csvfile))
        flipped = {key: convert_raw_data([x[key] for x in data]) for key in data[0]}
        flipped['Name'] = [x.split() for x in flipped['Name']]
        return flipped, self.get_schema()
