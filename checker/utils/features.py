from copy import deepcopy
from pydantic import BaseModel
from functools import partial
from typing import Dict
from typing import List
from typing import Optional


class Datapoint(BaseModel):
    id: Optional[str]
    label: int
    statement: str
    title: str
    subject: Optional[str]
    person: Optional[str]
    source: Optional[str]
    url: Optional[str]
    factchecker: Optional[str]


def construct_datapoint(input: str) -> Datapoint:
    return Datapoint(
        **{
            "statement": input,
        }
    )
