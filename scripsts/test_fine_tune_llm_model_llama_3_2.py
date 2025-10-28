import torch
import json
import os
import time

from safetensors.torch import load_file

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    # LlamaForCausalLM,
    BitsAndBytesConfig
    # LlamaTokenizer
)
from trl import setup_chat_format
from peft import PeftModel 

print("CUDA disponible :", torch.cuda.is_available())
print("Nombre de GPUs visibles :", torch.cuda.device_count())


for i in range(torch.cuda.device_count()):
    print(f"GPU {i} -> {torch.cuda.get_device_name(i)}")


# ==========================================================
# Fonction lecture fichier texte
# ==========================================================
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Dossiers de données
folder_path = "/home2020/home/icube/fghazoua/TetraProject/summurized_gpt"
output_folder = "/home2020/home/icube/fghazoua/TetraProject/outputs_summurize_text_llama3_2_gpt_LS40"
execution_time_file_path = os.path.join(output_folder, "execution_time.txt")

os.makedirs(output_folder, exist_ok=True)

max_chunk_size = 3000  # Maximum chunk size (en tokens)
execution_times = {}
   

# checkpoint_dir = "/home2020/home/icube/fghazoua/TetraProject/scripts/Qwen2.5-72B-Instruct-populate-ontology"
checkpoint_dir = "/home2020/home/icube/fghazoua/TetraProject/scripts/Llama-3.2-3B-populate-ontology"

#sd = load_file("/home2020/home/icube/fghazoua/TetraProject/scripts/llama-3-70b-populate-ontology/checkpoint-64/adapter_model.safetensors")
#for key, val in sd.items():
#    print(key, val.shape)

TUNED_MODEL_NAME = 'Llama-3.2-3B-populate-ontology' 
TUNED_BASE_DIRECTORY = '/home2020/home/icube/fghazoua/TetraProject/scripts'

# TUNED_MODEL_NAME = 'Qwen2.5-72B-Instruct-populate-ontology' 
# TUNED_BASE_DIRECTORY = '/home2020/home/icube/fghazoua/TetraProject/scripts'
tuned_model_directory = f'{TUNED_BASE_DIRECTORY}/{TUNED_MODEL_NAME}'

# MODEL_NAME = 'Qwen/Qwen2.5-72B-Instruct' #'meta-llama/Llama-3.3-70B-Instruct' #'meta-llama/Llama-3.2-3B-Instruct'
MODEL_NAME = 'meta-llama/Llama-3.2-3B'
BASE_DIRECTORY = '/home2020/home/icube/fghazoua/TetraProject/models'
base_model_directory = f'{BASE_DIRECTORY}/{MODEL_NAME}'

# 
adapters_name = tuned_model_directory

# attn_implementation = "eager" 
attn_implementation = "flash_attention_2" 

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Process started....")
start_time = time.time()

print(f"Starting to load the model {TUNED_MODEL_NAME} into memory")


# Reload tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(base_model_directory)

base_model_reload = AutoModelForCausalLM.from_pretrained(
    base_model_directory,
    return_dict=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
    # quantization_config=bnb_config,
    # device_map='auto',
    # attn_implementation=attn_implementation
)

base_model_reload, tokenizer = setup_chat_format(base_model_reload, tokenizer)

# merge adopter with base model
model = PeftModel.from_pretrained(
	base_model_reload, 
	checkpoint_dir
	# model_id=checkpoint_dir,
	#peft_config=bnb_config,
	#device_map='auto',
	# attn_implementation=attn_implementation
)

model = model.merge_and_unload()

# Load tokenizer
# tokenizer = LlamaTokenizer.from_pretrained(checkpoint_dir)

# Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_directory)

# if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
#     tokenizer.chat_template = None  # Reset the chat template

# model, tokenizer = setup_chat_format(model, tokenizer)

# testing the new model

# messages = [
#     {
#         "role": "system",
#         "content": "Translate the user text into an TTL graph based on the TetraOnto ontology."
#     },
#     {
#        "role": "user",
#        "content": """A restoration project in the Santa Ynez River Watershed, Central California, retrofitted a concrete and rock grade control apron downstream of the Highway 1 bridge with a step-pool fishway to improve steelhead passage at low flows. The 40 ft (12.2 m) apron previously blocked fish movement under 20 cfs (0.57 cms) flows. The project added three small step-pools and a 30 ft (9.1 m) concrete sidewall to direct low flows into the pools, improving passage conditions while preserving the structure’s function. Completed in January 2002 for $87,000, it was funded by the Cachuma Conservation and Release Board, Entrix, Inc., Allen and Robert Larson, and the California Coastal Conservancy."""
#    }
#]

MODEL_NAME = 'Llama-3.2-3B'
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)

    if not file_name.endswith(".txt"):
        continue  # Skip fichiers non-texte

    user_text = read_text_file(file_path)
    print(f"\n=== Traitement du fichier '{file_name}' avec le modèle '{MODEL_NAME}' ===")

    messages = [
        {"role": "system", "content": "Translate the user text into an TTL graph based on the TetraOnto ontology."},
        {"role": "user", "content": user_text}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, 
                                       add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors='pt', padding=True, 
                   truncation=True).to(model.device)#.to("cuda")

    # Mesure du temps d'exécution
    start_time = time.time()

    try:
        outputs = model.generate(
            **inputs,
            #max_length=8192,
	    max_new_tokens=1280,
	    do_sample=True,
	    temperature=0.7,
	    top_p=0.9,
	    #eos_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,  # Stopper la génération ici
    	    pad_token_id=tokenizer.eos_token_id,  # Évite les warnings de padding
    	   repetition_penalty=1.2,               # Réduit les répétitions
            num_return_sequences=1
        )
    except Exception as e:
        print(f"Erreur pendant la génération pour le fichier {file_name} : {e}")
        raise

    # outputs = model.generate(**inputs, max_length=8192, 
    #                     num_return_sequences=1)

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Calcul du temps d'exécution
    end_time = time.time()
    execution_time = end_time - start_time
    minutes, seconds = divmod(execution_time, 60)
    formatted_time = f"{int(minutes)} minutes and {seconds:.2f} seconds"
    print(f"Execution time: {formatted_time}")

    # Sauvegarde du temps d'exécution
    with open(execution_time_file_path, "a", encoding="utf-8") as time_file: 
        time_file.write(f"Execution time for model '{MODEL_NAME}' on file '{file_name}': {formatted_time}\n")

    # Sauvegarde de la sortie TTL
    output_file_name = f"{os.path.splitext(file_name)[0]}_{MODEL_NAME.replace('-', '_')}_{1280}_{torch.cuda.get_device_name(0)}.ttl"
    output_file_path = os.path.join(output_folder, output_file_name)

    # Découper pour isoler la partie TTL après "assistant"
    if "assistant" in text:
        turtle_data = text.split("assistant", 1)[1]
    else:
        turtle_data = text

    with open(output_file_path, "w", encoding="utf-8") as output_file: 
        output_file.write(turtle_data)

    print(f"Résultat sauvegardé dans '{output_file_path}'.")

    # print(text.split("assistant")[1])

model.save_pretrained("/home2020/home/icube/fghazoua/TetraProject/llama-3.2-3B-populate-ontology")
tokenizer.save_pretrained("/home2020/home/icube/fghazoua/TetraProject/llama-3.2-3B-populate-ontology")

# end_time = time.time()
# execution_time = end_time - start_time
# minutes, seconds = divmod(execution_time, 60)
# formatted_time = f"{int(minutes)} minutes and {seconds:.2f} seconds"
# print(f"Execution time for model '{MODEL_NAME}' is: {formatted_time}")
print("\nProcessing complete for all files and models.")
