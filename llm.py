from ollama import Client
""" from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template """
from system_prompt import system_prompt

client = Client(
    host="http://192.168.0.20:11434",
    headers={
        "Connection": "keep-alive",
        "Keep-Alive": "timeout=5, max=120",
    }
)

def generate(model: str, system_prompt: str, question: str) -> str:
    response = client.chat(model=model, messages=[{
            "role": "system",
            "content": system_prompt,
        },
        {
            'role': 'user',
            'content': question,
        },
    ])
    return response['message']['content']

""" def load_unsloth_model(model_name: str) -> any:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def generate_unsloth(model: any, tokenizer: any, system_prompt: str, question: str) -> str:
    messages=[{
            "role": "system",
            "content": system_prompt,
        },
        {
            'role': 'user',
            'content': question,
        },
    ]
    inputs = tokenizer.apply_chat_template(messages, tokenize = True, add_generation_prompt = True, return_tensors = "pt").to("cuda")
    response = model.generate(input_ids = inputs, max_new_tokens = 1024, use_cache = True)
    return response

model, tokenizer = load_unsloth_model("./llama3.2_teacher")
response = generate_unsloth(model, tokenizer, system_prompt, "donne moi la réponse")
print(response) """

