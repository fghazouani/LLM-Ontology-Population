""" This script is used to test the fine-tuned Llama-3.2-3B model for ontology population from user text input files. """

import torch
import json
import os

from safetensors.torch import load_file

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)
from trl import setup_chat_format
from peft import PeftModel 


def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# path to the folder containing input text files
folder_path = "./.../test_data"

# define path to the output folder
output_folder = ".../outputs_results_llama3_2-3b"
os.makedirs(output_folder, exist_ok=True)

# get the path were the fine-tuned model (Llama-3.2-3B-populate-ontology) is saved
checkpoint_dir = "/{path to the saved fine-tuned model}/Llama-3.2-3B-populate-ontology"

TUNED_MODEL_NAME = 'Llama-3.2-3B-populate-ontology' 
TUNED_BASE_DIRECTORY = '{path to the fine-tuned model}'
tuned_model_directory = f'{TUNED_BASE_DIRECTORY}/{TUNED_MODEL_NAME}'

# path to the base model (Llama-3.2-3B)
MODEL_NAME = 'meta-llama/Llama-3.2-3B'
BASE_DIRECTORY = '{path to the base model (Llama-3.2-3B)}'
base_model_directory = f'{BASE_DIRECTORY}/{MODEL_NAME}'

# 
adapters_name = tuned_model_directory

attn_implementation = "flash_attention_2" 

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


# Reload tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(base_model_directory)

base_model_reload = AutoModelForCausalLM.from_pretrained(
    base_model_directory,
    return_dict=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

base_model_reload, tokenizer = setup_chat_format(base_model_reload, tokenizer)

# merge adopter with base model
model = PeftModel.from_pretrained(
	base_model_reload, 
	checkpoint_dir
	peft_config=bnb_config,
)

model = model.merge_and_unload()

MODEL_NAME = 'Llama-3.2-3B'
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)

    if not file_name.endswith(".txt"):
        continue  

    user_text = read_text_file(file_path)

    messages = [
        {"role": "system", "content": "Translate the user text into an TTL graph based on the TetraOnto ontology."},
        {"role": "user", "content": user_text}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, 
                                       add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors='pt', padding=True, 
                   truncation=True).to(model.device)#.to("cuda")

    # Run time measurement
    start_time = time.time()

    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=1280,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,  
    	    pad_token_id=tokenizer.eos_token_id,  
    	    repetition_penalty=1.2,              
            num_return_sequences=1
        )
    except Exception as e:
        print(f"Erreur generation for the file: {file_name} : {e}")
        raise


    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # create output file for result
    output_file_name = f"{os.path.splitext(file_name)[0]}_{MODEL_NAME.replace('-', '_')}_{torch.cuda.get_device_name(0)}.ttl"
    output_file_path = os.path.join(output_folder, output_file_name)

    # Cut to isolate the TTL part after “assistant”
    if "assistant" in text:
        turtle_data = text.split("assistant", 1)[1]
    else:
        turtle_data = text

    # save extracted TTL triplets
    with open(output_file_path, "w", encoding="utf-8") as output_file: 
        output_file.write(turtle_data)

print("\nProcessing complete for all files and models.")
