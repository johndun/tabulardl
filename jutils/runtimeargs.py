import argparse
import os
import pickle
from dataclasses import dataclass, fields
from datetime import datetime


@dataclass
class RuntimeArgs:
    """Runtime arguments base class.

    * To use, extend this class with additional attributes. Then `args = Args().parse_args()`
    * `parse_args` will throw an error unless all fields are strings, ints, bools, or floats

    Attributes:
        run_date: ISO format string representation of the run date.
        artifacts_basepath: Local path where artifacts are to be saved.
    """
    run_date: str = str(datetime.now())
    artifacts_basepath: str = '.'

    def __post_init__(self):
        os.makedirs(self.artifacts_basepath, exist_ok=True)

    def parse_args(self):
        """Assign attributes from command line argument values."""
        parser = argparse.ArgumentParser()
        for field in fields(self):
            if type(field.default) in (int, float, str):
                parser.add_argument(
                    f'--{field.name}', type=type(field.default), default=field.default
                )
            elif type(field.default) == bool and field.default is False:
                parser.add_argument(
                    f'--{field.name}', dest=field.name, default=False, action='store_true'
                )

        self.__init__(**vars(parser.parse_args()))
        return self

    def get_run_date(self) -> datetime:
        """Returns the run_date attribute as a datetime object."""
        return datetime.fromisoformat(self.run_date)

    def save(self):
        pickle.dump(self, open(f'{self.artifacts_basepath}/args.pickle', 'wb'))

    def load(self):
        return pickle.load(open(f'{self.artifacts_basepath}/args.pickle', 'rb'))
