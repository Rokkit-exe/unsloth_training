from models.exercice import Exercice
from pydantic import BaseModel

class Dataset(BaseModel): list[Exercice]

