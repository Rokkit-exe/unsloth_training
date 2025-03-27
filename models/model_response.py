from models.qa import QA
from pydantic import BaseModel

class ModelResponse(BaseModel):
    model: str = ''
    response: str = ''