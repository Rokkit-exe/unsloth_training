import requests
import json
import os
from models.exercice import Exercice
from models.qa import QA
from system_prompt import system_prompt_etudiant, system_prompt_étudiant_malveillant, system_prompt
from ollama import chat, ChatResponse

question_dir = './prog_1/questions'

dataset_file = './data/dataset.json'

def generate_questions(system_prompt, enonce, ebauche):
    response: ChatResponse = chat(model='gemma3:12b', messages=[{
            "role": "system",
            "content": system_prompt,
        },
        {
            'role': 'user',
            'content': f"# Énoncé de l'exercice: {enonce}\n# Ébauche de l'exercice: {ebauche}",
        },
    ])
    return response['message']['content']
    

def generate_reponse(system_prompt, enonce, ebauche, question):
    response: ChatResponse = chat(model='gemma3:12b', messages=[{
            "role": "system",
            "content": system_prompt,
        },
        {
            'role': 'user',
            'content': f"# Énoncé de l'exercice: {enonce}\n# Ébauche de l'exercice: {ebauche}\n# Question: {question}",
        },
    ])
    return response['message']['content']

def create_exercices(dir):
    dataset = list[Exercice]()
    if not os.path.exists(dir) and not os.path.isdir(dir):
        return None
    for root, dirs, files in os.walk(dir):
        exercice = Exercice()
        for file in files:
            if file == 'info.yml':
                data = read_file(root, file)
                exercice.enonce = json.dumps(data)
            if file.endswith('.py') or file.endswith('.java') or file.endswith('.js') or file.endswith('.kt'):
                if exercice.enonce == '':
                    data = read_file(root, file)
                    exercice.ebauche = json.dumps(data)
            if exercice.enonce != '' and exercice.ebauche != '':
                dataset.append(exercice)
    return dataset

def generate_qa(dataset: list[Exercice]):
    count = 0
    for exercice in dataset:
        count += 1
        if exercice.qa != []:
            print(f"QA already generated at {count}/{len(dataset)}")
            continue
        print(f"Generating QA for exercice {count}/{len(dataset)}")
        try:
            qa_list = generate_questions(system_prompt_etudiant, exercice.enonce, exercice.ebauche)
            qa_list = strip_markdown(qa_list)
            qa_list = json.loads(qa_list)
            qa_list = [QA(**qa) for qa in qa_list]
            exercice.qa = qa_list
            print(f"good questions generated")
            write_dataset(dataset, dataset_file)
        except Exception as e:
            print(e)
            continue
        try:
            qa_list = generate_questions(system_prompt_étudiant_malveillant, exercice.enonce, exercice.ebauche)
            qa_list = strip_markdown(qa_list)
            qa_list = json.loads(qa_list)
            qa_list = [QA(**qa) for qa in qa_list]
            for qa in qa_list:
                exercice.qa.append(qa)
            print(f"bad questions generated")
            write_dataset(dataset, dataset_file)
        except Exception as e:
            print(e)
            continue
        try:
            for qa in exercice.qa:
                question = qa.question
                reponse = generate_reponse(system_prompt, exercice.enonce, exercice.ebauche, question)
                qa.reponse = reponse
            print(f"Responses generated")
            write_dataset(dataset, dataset_file)
        except Exception as e:
            print(e)
            continue
        write_dataset(dataset, dataset_file)
    return dataset
    


def create_qa(dataset: list[Exercice]):
    for exercice in dataset:
        exercice.qa = list[QA]()
        for qa in exercice.enonce['qa']:
            exercice.qa.append(QA(**qa))
    return dataset

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

def strip_markdown(text: str):
    return text.replace('```json', '').replace('```', '')

""" print('Create dataset')
dataset = create_exercices(question_dir)
print("writing dataset")
write_dataset(dataset, dataset_file) """
print("reading dataset")
dataset = read_dataset(dataset_file)
print("generating qa")
generate_qa(dataset)






