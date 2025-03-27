from models.question import Question
from pydantic import BaseModel

class ModelResponseExercice(BaseModel):
    enonce: str = ''
    ebauche: str = ''
    questions: list[Question] = []
