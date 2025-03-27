from unsloth import FastLanguageModel

model_name = "unsloth/Llama-3.2-3B-Instruct"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name, 
    max_seq_length=2048, 
    dtype=None,
    load_in_4bit=True
)

model.load_adapter("./outputs/checkpoint-60")

model.save_pretrained("llama3.2_teacher", tokenizer=tokenizer, quantization_method = "q4_k_m")