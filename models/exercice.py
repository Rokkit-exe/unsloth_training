from pydantic import BaseModel
from models.qa import QA


class Exercice(BaseModel):
    enonce: str = ''
    ebauche: str = ''
    qa: list[QA] = []
    