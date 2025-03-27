from models.exercice import Exercice
from models.model_response_exercice import ModelResponseExercice
from models.qa import QA
import json
import os

def write_dataset(dataset: list, dataset_file: str):
    serialized_dataset = [data.model_dump() for data in dataset]
    write(dataset_file, json.dumps(serialized_dataset, indent=4))

def read(dir, file):
    filepath = os.path.join(dir, file)
    with open(filepath, 'r', encoding="utf-8") as f:
        return f.read()
    
def write(filepath, content):
    with open(filepath, 'w') as f:
        f.write(content)

def load_json(filepath: str):
    with open(filepath, 'r') as f:
        return json.loads(f.read())
    
def load_dataset(filepath: str) -> list[Exercice]:
    json_data = load_json(filepath)
    dataset = list[Exercice]()
    for exercice in json_data:
        dataset.append(Exercice(**exercice))
    return dataset

def load_dataset_response(filepath: str) -> list[ModelResponseExercice]:
    json_data = load_json(filepath)
    dataset = list[ModelResponseExercice]()
    for exercice in json_data:
        dataset.append(ModelResponseExercice(**exercice))
    return dataset