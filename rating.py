import json
from models.note import Note
from models.model_response_exercice import ModelResponseExercice
from system_prompt import system_prompt_rating
from file import load_dataset_response
from llm import generate

model = "gemma3:12b"
dataset_file = "./data/dataset_mini.json"
output_file = "./data/dataset_mini_responses.json"

dataset = load_dataset_response(output_file)

def strip_markdown(text: str):
    return text.replace('```json', '').replace('```', '')

def generate_rating(dataset: list[ModelResponseExercice])-> dict:
    score = {
        "llama3.2:3b": 0,
        "mistral:7b": 0,
        "gemma3:4b": 0,
        "gemma3:12b": 0,
        "qwen2.5:3b": 0,
        "qwen2.5:7b": 0,
    }
    for e_index, exercice in enumerate(dataset):
        for q_index, question in enumerate(exercice.questions):
            responses = question.responses_to_json()
            content = f"### Énoncé\n{exercice.enonce}\n### Ébauche\n{exercice.ebauche}\n### Question\n{question.question}\n### Réponses\n{responses}"
            print(f"Generating response for exercice {e_index}/{len(dataset)} question {q_index}/{len(exercice.questions)}")
            response = generate(model, system_prompt_rating, content)
            try:
                response = strip_markdown(response)
                response = json.loads(response)
                note_list = [Note(**note) for note in response]
            except Exception as e:
                print(response)
                print(e)
                continue
            for note in note_list:
                print(f"model: {note.model} note: {note.note}")
                score[note.model] += note.note
    return score






