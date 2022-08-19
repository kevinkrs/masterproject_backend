import csv
from typing import List

from checker.utils.features import Datapoint


def read_csv_data(datapath: str) -> List[Datapoint]:
    with open(datapath) as f:
        datapoints = csv.load(f)
        return [Datapoint(**point) for point in datapoints]
