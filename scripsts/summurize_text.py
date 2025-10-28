import json
import os
import time
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


from trl import setup_chat_format

from peft import PeftModel    


MODEL_NAME = 'Qwen/Qwen2.5-72B-Instruct' # 'meta-llama/Llama-3.3-70B-Instruct' #'meta-llama/Llama-3.2-3B-Instruct'
BASE_DIRECTORY = '/home2020/home/icube/fghazoua/TetraProject/models'
model_directory = f'{BASE_DIRECTORY}/{MODEL_NAME}'

# 
# adapters_name = tuned_model_directory

attn_implementation = "eager" 
# attn_implementation = "flash_attention_2" # need pip install -qqq flash-attn

print("Process started....")
start_time = time.time()

print(f"Starting to load the model {MODEL_NAME} into memory")

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load the base model
model = AutoModelForCausalLM.from_pretrained(
    model_directory,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation=attn_implementation
)


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_directory)

if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
    tokenizer.chat_template = None  # Reset the chat template

model, tokenizer = setup_chat_format(model, tokenizer)

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Parameters

folder_path = "/home2020/home/icube/fghazoua/TetraProject/base_test"  # Folder containing input text files

max_chunk_size = 3000  # Maximum chunk size (in tokens)

output_folder = "/home2020/home/icube/fghazoua/TetraProject/summurize_responses"  # Folder to save outputs
execution_time_file_path = "/home2020/home/icube/fghazoua/TetraProject/summurize_responses/execution_time.txt"
# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

execution_times = {}  # Dictionary to store execution times 

# Process each text file in the folder
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)

    # Skip non-text files
    if not file_name.endswith(".txt"):
        continue

    # Read the content of the file
    user_text = read_text_file(file_path)


    print(f"Processing file '{file_name}' with model '{MODEL_NAME}'...")

    messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant specialized in summarizing semi-structured ecological restoration project descriptions. Your task is to read the input data and generate a clear, concise, and informative paragraph in French or in English (depending of the text input), summarizing the key information such as location, restoration type, ecological goal, stakeholders, technical elements, monitoring activities, and funding. The output should be written in natural French or Englishs, suitable for inclusion in reports or databases."
    },
    {
        "role": "user",
        "content": user_text
    }
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, 
                                        add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors='pt', padding=True, 
                    truncation=True).to("cuda")

    # Start timing the execution
    start_time = time.time()

    outputs = model.generate(**inputs, max_length=2048, 
                            num_return_sequences=1)

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(text)
    print("-------------------------------------->>>>")
    print(text.split("assistant")[1])



    # Stop timing the execution
    end_time = time.time()

    # Calculate execution time
    execution_time = end_time - start_time
    # execution_times[(file_name, model)] = execution_time

    minutes, seconds = divmod(execution_time, 60)
    formatted_time = f"{int(minutes)} minutes and {seconds:.2f} seconds"
        
    print(f"Execution time for model '{MODEL_NAME}' on file '{file_name}': {formatted_time}")

    with open(execution_time_file_path, "a", encoding="utf-8") as time_file: 
        # time_file.write(f"Execution time for model '{model}' on file '{file_name}': {execution_time:.2f} seconds\n")
        time_file.write(f"Execution time for model '{MODEL_NAME}' on file '{file_name}': {formatted_time}\n")


    output_file_name = f"{os.path.splitext(file_name)[0]}.txt"
    output_file_path = os.path.join(output_folder, output_file_name)

    turtle_data = text.split("assistant")[2].strip() # [2] because there is two terms 'assistant' in the output
    
    with open(output_file_path, "w", encoding="utf-8") as output_file: 
        output_file.write(turtle_data)

    # print(f"Saved responses to '{output_file_path}'.")
    

print("Processing complete for all files and models.")

