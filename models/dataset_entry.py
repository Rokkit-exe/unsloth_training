from pydantic import BaseModel

class DatasetEntry(BaseModel):
    enonce: str = ''
    ebauche: str = ''
    question: str = ''
    reponse: str = ''