import json
import os
from models.exercice import Exercice
from models.qa import QA
from models.model_response import ModelResponse
from models.question import Question
from models.model_response_exercice import ModelResponseExercice
from system_prompt import system_prompt
from file import write_dataset, load_dataset, load_dataset_response
from llm import generate

models = [
    "llama3.2:3b",
    "mistral:7b",
    "gemma3:4b"
    "gemma3:12b",
    "qwen2.5:3b",
    "qwen2.5:7b",
]

models = [
    "llama3.2_t"
]

dataset_file = "./data/dataset_20x20.json"
output_file = "./data/dataset_20x20_reponses.json"


def create_dataset(dataset: list[Exercice], models: list[str], output_file: str):
    response_dataset = list[ModelResponseExercice]()
    for e_index, exercice in enumerate(dataset):
        model_reponse_exercice = ModelResponseExercice(
            enonce=exercice.enonce,
            ebauche=exercice.ebauche,
            questions=list[Question]()
        )
        for q_index, qa in enumerate(exercice.qa):
            question = Question(
                question = qa.question,
                responses = list[ModelResponse]()
            )
            for model in models:
                model_reponse = ModelResponse(
                    model=model,
                    response=""
                )
                question.responses.append(model_reponse)
            model_reponse_exercice.questions.append(question)
        response_dataset.append(model_reponse_exercice)
    write_dataset(response_dataset, output_file)

def generate_dataset_reponse(dataset: list[Exercice], models: list[str], output_file: str):
    for m_index, model in enumerate(models):
        for e_index, exercice in enumerate(dataset):
            for q_index, qa in enumerate(exercice.qa):
                if response_dataset[e_index].questions[q_index].responses[m_index].response != "":
                    continue
                print(f"Generating response for exercice {e_index}/{len(dataset)} question {q_index}/{len(exercice.qa)} model {m_index}/{len(models)}")
                content = f"### Énoncé\n{exercice.enonce}\n### Ébauche\n{exercice.ebauche}\n### Question\n{qa.question}"
                response = generate(model, system_prompt, content)
                response_dataset[e_index].questions[q_index].responses[m_index].response = response
            print(f"Writing dataset to {output_file}")
            write_dataset(response_dataset, output_file)

def remove_teacher(dataset: list[ModelResponseExercice]):
    for exercice in dataset:
        for question in exercice.questions:
            question.responses.pop(-1)
    return dataset

def add_new_model(dataset: list[ModelResponseExercice], model: str):
    for exercice in dataset:
        for question in exercice.questions:
            model_reponse = ModelResponse(
                model=model,
                response=""
            )
            question.responses.append(model_reponse)
    return dataset

dataset = load_dataset(dataset_file)
output_file = "./data/dataset_20x20_response_teacher.json"
create_dataset(dataset, models, output_file)
response_dataset = load_dataset_response(output_file)
generate_dataset_reponse(dataset, models, output_file)
write_dataset(dataset, output_file)





