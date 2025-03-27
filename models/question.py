from models.model_response import ModelResponse
from pydantic import BaseModel


class Question(BaseModel):
    question: str = ''
    responses: list[ModelResponse] = []

    def responses_to_json(self) -> str:
        response_str = "["
        for response in self.responses:
            response_str += response.model_dump_json()
        response_str += "]" if len(self.responses) > 0 else ""
        return response_str

