import torch
import json
import os
import time

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from trl import setup_chat_format

print("CUDA disponible :", torch.cuda.is_available())
print("Nom du GPU :", torch.cuda.get_device_name(0))
print("Version CUDA détectée :", torch.version.cuda)
print("Capacité de calcul (compute capability) :", torch.cuda.get_device_capability(0))

MODEL_NAME = 'llama-3-70b-populate-ontology_2' #'llama-3-70b-populate-ontology' #'llama-3-70b-populate-ontology_2' #'llama-3-70b-populate-ontology (Qween)' #
BASE_DIRECTORY = '/home2020/home/icube/fghazoua/TetraProject'

model_directory = f'{BASE_DIRECTORY}/{MODEL_NAME}'

torch_dtype = torch.float16
# attn_implementation = "eager" 
attn_implementation = "flash_attention_2"

def load_model(model_directory=model_directory, torch_dtype=torch_dtype, attn_implementation=attn_implementation):
    # QLoRA config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
    )

    # print("Process started....")
    # start_time = time.time()
    # Load model
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

    return model, tokenizer


def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Parameters

# folder_path = "/home2020/home/icube/fghazoua/TetraProject/base_test"  # Folder containing input text files
# folder_path = "/home2020/home/icube/fghazoua/TetraProject/summurize_responses"
folder_path = "/home2020/home/icube/fghazoua/TetraProject/summurized_gpt"

max_chunk_size = 3000  # Maximum chunk size (in tokens)

# output_folder = "/home2020/home/icube/fghazoua/TetraProject/output_responses_Qwen"  # Folder to save outputs
# output_folder = "/home2020/home/icube/fghazoua/TetraProject/outputs_summurize_text_llama3_3"
output_folder = "/home2020/home/icube/fghazoua/TetraProject/outputs_summurize_text_llama3_3_gpt_pop_Llama3_3_1024"
execution_time_file_path = "/home2020/home/icube/fghazoua/TetraProject/outputs_summurize_text_llama3_3_gpt_pop_Llama3_3_1024/execution_time.txt"
# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

execution_times = {}  # Dictionary to store execution times 

model, tokenizer = load_model(model_directory, torch_dtype, attn_implementation)

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
            "content": "Translate the user text into an TTL graph based on the TetraOnto ontology."
        },
        {
            "role": "user",
            "content":user_text 
        }
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, 
                                        add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors='pt', padding=True, 
                    truncation=True).to(model.device)#.to("cuda")

    # Start timing the execution
    start_time = time.time()

    outputs = model.generate(**inputs, #max_length=1024, 
                            #num_return_sequences=1,
			    max_new_tokens=1024, #1280,
			    do_sample=True,
			    temperature=0.7,
			    top_p=0.9,
			    #eos_token_id=tokenizer.eos_token_id,
			    eos_token_id=tokenizer.eos_token_id,  # Stopper la génération ici
		    	    pad_token_id=tokenizer.eos_token_id,  # Évite les warnings de padding
		    	    repetition_penalty=1.2,               # Réduit les répétitions
			    num_return_sequences=1)

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # max_length=16200

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


    output_file_name = f"{os.path.splitext(file_name)[0]}_{MODEL_NAME.replace('-', '_')}_1024.ttl"
    output_file_path = os.path.join(output_folder, output_file_name)

    turtle_data = text.split("assistant")[1]
    with open(output_file_path, "w", encoding="utf-8") as output_file: 
        output_file.write(turtle_data)

    # print(f"Saved responses to '{output_file_path}'.")
    

print("Processing complete for all files and models.")

