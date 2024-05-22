# pip install accelerate
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name_or_path = "google/gemma-2b"
# model_name_or_path = "internlm/internlm2-1_8b"
# model_name_or_path = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)

input_text = "My name is Julien and I like to"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_length=100, return_dict_in_generate=True, output_hidden_states=True)
print(outputs.hidden_states[-1][-1].dtype)
print(tokenizer.decode(outputs.sequences[0]))
