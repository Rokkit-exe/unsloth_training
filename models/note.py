from pydantic import BaseModel

class Note(BaseModel):
    model: str = ''
    note: int = 0

