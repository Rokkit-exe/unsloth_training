from pydantic import BaseModel

class QA(BaseModel):
    question: str = ''
    reponse: str = ''