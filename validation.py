from ollama import chat, ChatResponse
from models.exercice import Exercice
from models.qa import QA
from system_prompt import system_prompt_validation_contexte, system_prompt_validation_reponse, system_prompt_reformulation
import json
import os

def generate_context_validation(system_prompt,enonce, ebauche, question, reponse):
    response: ChatResponse = chat(model='gemma3:12b', messages=[{
            "role": "system",
            "content": system_prompt,
        },
        {
            'role': 'user',
            'content': f"# Énoncé de l'exercice: {enonce}\n#Ébauche de l'exercice: {ebauche}\n# Question: {question}\n# Réponse: {reponse}",
        },
    ])
    return response['message']['content']

def generate_reponse_validation(system_prompt, enonce, ebauche, question, reponse):
    response: ChatResponse = chat(model='gemma3:12b', messages=[{
            "role": "system",
            "content": system_prompt,
        },
        {
            'role': 'user',
            'content': f"# Énoncé de l'exercice: {enonce}\n#Ébauche de l'exercice: {ebauche}\n# Question: {question}\n# Réponse: {reponse}",
        },
    ])
    return response['message']['content']

def generate_reformulation_reponse(system_prompt, enonce, ebauche, question, reponse):
    response: ChatResponse = chat(model='gemma3:12b', messages=[{
            "role": "system",
            "content": system_prompt,
        },
        {
            'role': 'user',
            'content': f"# Énoncé de l'exercice: {enonce}\n#Ébauche de l'exercice: {ebauche}\n# Question: {question}\n# Réponse: {reponse}",
        },
    ])
    return response['message']['content']

def write_dataset(dataset: list[Exercice], dataset_file: str):
    serialized_dataset = [exercice.model_dump() for exercice in dataset]
    write_file(dataset_file, json.dumps(serialized_dataset, indent=4))

def read_file(dir, file):
    filepath = os.path.join(dir, file)
    with open(filepath, 'r', encoding="utf-8") as f:
        return f.read()
    
def write_file(filepath, content):
    with open(filepath, 'w') as f:
        f.write(content)

def load_json(dataset_file):
    with open(dataset_file, 'r') as f:
        return json.loads(f.read())
    
def read_dataset(dataset_file):
    json_data = load_json(dataset_file)
    dataset = list[Exercice]()
    for exercice in json_data:
        dataset.append(Exercice(**exercice))
    return dataset

def validation_pipeline(dataset: list[Exercice]):
    for exercice in dataset:
        for qa in exercice.qa:
            reponse_validation = generate_reponse_validation(system_prompt_validation_reponse, exercice.enonce, exercice.ebauche, qa.question, qa.reponse)
            if reponse_validation == "oui":
                print("réponse donne le code de la solution de l'exercice")
                reponse_reformuler = generate_reformulation_reponse(system_prompt_reformulation, exercice.enonce, exercice.ebauche, qa.question, qa.reponse)
                print(f"REFORMULATION: {reponse_reformuler}")
                validation = input("Est-ce que vous validez? (y/N): ")
                if validation == "y":
                    print("réponse reformuler validé")
                else:
                    print("réponse reformuler non validé")
            elif reponse_validation == "non":
                print("réponse ne donne pas le code de la solution de l'exercice")
            else:
                print("réponse non valide")
    return dataset

def strip_markdown(text: str):
    return text.replace('```json', '').replace('```', '')

def validation_context_pipeline(dataset: list[Exercice]):
    for exercice in dataset:
        for qa in exercice.qa:
            response = generate_context_validation(system_prompt_validation_contexte, exercice.enonce, exercice.ebauche, qa.question, qa.reponse)
            print(f"Response: {response}")
            try:
                response_trimed = strip_markdown(response)
                json_data = json.loads(response_trimed)
                context_validation = QA(**json_data)
            except Exception as e:
                print(e)
                continue
            if context_validation.question == "non" or context_validation.reponse == "non":
                print("Context non valide")
                print(f"Question: {qa.question}")
                print(f"Réponse: {qa.reponse}")
                validation = input("Est-ce que vous validez? (y/N): ")
                if validation == "y":
                    print("Context validé")
                else:
                    print("Context non validé")
                    response = generate_reformulation_reponse(system_prompt_reformulation, exercice.enonce, exercice.ebauche, qa.question, qa.reponse)
                    print(f"REFORMULATION: {response}")
                    validation = input("Est-ce que vous validez? (y/N): ")
    return dataset

dataset = read_dataset("./data/dataset_mini.json")
dataset = validation_context_pipeline(dataset)

