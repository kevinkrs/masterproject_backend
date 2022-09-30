from pydantic import BaseModel
from typing import Optional


class DataModel(BaseModel):
    text: str
    statementdate: str
    author: Optional[str]
    source: Optional[str]


class TrainModel(BaseModel):
    text: str
    statementdate: str
    author: Optional[str]
    source: Optional[str]
    results: object
