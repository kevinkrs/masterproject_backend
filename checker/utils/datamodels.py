from pydantic import BaseModel
from typing import Optional


class DataModel(BaseModel):
    text: str
    statementdate: str
    author: Optional[str]
    source: Optional[str]
