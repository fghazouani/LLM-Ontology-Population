import os
import gc
import time
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import setup_chat_format

# ==========================================================
# 0) Debug option - activer pour avoir une stacktrace précise
# ==========================================================
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# ==========================================================
# 1) Infos CUDA et GPU
# ==========================================================
print("CUDA disponible :", torch.cuda.is_available())
print("Nombre de GPUs visibles :", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i} -> {torch.cuda.get_device_name(i)}")

print("Version CUDA détectée :", torch.version.cuda)
print("Capacité de calcul (compute capability) :", torch.cuda.get_device_capability(0))

# ==========================================================
# 2) Paramètres principaux
# ==========================================================
MODEL_NAME = 'llama-3-70b-populate-ontology_2'
BASE_DIRECTORY = '/home2020/home/icube/fghazoua/TetraProject'
model_directory = f'{BASE_DIRECTORY}/{MODEL_NAME}'

torch_dtype = torch.float16

# Si flash attention pose problème, repasse en "eager"
attn_implementation = "flash_attention_2"  # ou "eager"

# Dossiers de données
folder_path = "/home2020/home/icube/fghazoua/TetraProject/summurized_gpt"
output_folder = "/home2020/home/icube/fghazoua/TetraProject/outputs_summurize_text_llama3_3_gpt_pop_2"
execution_time_file_path = os.path.join(output_folder, "execution_time.txt")

os.makedirs(output_folder, exist_ok=True)

max_chunk_size = 3000  # Maximum chunk size (en tokens)
execution_times = {}

# ==========================================================
# 3) Fonction pour charger le modèle et le tokenizer
# ==========================================================
def load_model(model_directory, torch_dtype, attn_implementation):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
    )

    print("\nChargement du modèle...")
    model = AutoModelForCausalLM.from_pretrained(
        model_directory,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=attn_implementation
    )
    print("Modèle chargé avec succès.")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
        tokenizer.chat_template = None  # Reset chat template si besoin

    model, tokenizer = setup_chat_format(model, tokenizer)

    # Affichage de la répartition du modèle sur les devices
    print("\n===== Répartition du modèle (hf_device_map) =====")
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        for module, device in model.hf_device_map.items():
            print(f"{module} -> {device}")
    else:
        print("Pas de hf_device_map détecté")

    return model, tokenizer

# ==========================================================
# 4) Fonction lecture fichier texte
# ==========================================================
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# ==========================================================
# 5) Chargement du modèle
# ==========================================================
model, tokenizer = load_model(model_directory, torch_dtype, attn_implementation)

# Déterminer le device principal pour les inputs
if hasattr(model, "hf_device_map") and model.hf_device_map:
    first_device = list(model.hf_device_map.values())[0]
else:
    first_device = "cuda:0"
print(f"\nLes inputs seront déplacés vers : {first_device}\n")

# ==========================================================
# 6) Traitement des fichiers
# ==========================================================
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

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # IMPORTANT : inputs envoyés sur le même device que le modèle
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
    inputs = {k: v.to(first_device) for k, v in inputs.items() if v is not None}

    # Vider la mémoire avant de générer
    gc.collect()
    torch.cuda.empty_cache()

    # Mesure du temps d'exécution
    start_time = time.time()

    try:
        outputs = model.generate(
            **inputs,
            max_length=8192,
            num_return_sequences=1
        )
    except Exception as e:
        print(f"Erreur pendant la génération pour le fichier {file_name} : {e}")
        raise

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
    output_file_name = f"{os.path.splitext(file_name)[0]}_{MODEL_NAME.replace('-', '_')}.ttl"
    output_file_path = os.path.join(output_folder, output_file_name)

    # Découper pour isoler la partie TTL après "assistant"
    if "assistant" in text:
        turtle_data = text.split("assistant", 1)[1]
    else:
        turtle_data = text

    with open(output_file_path, "w", encoding="utf-8") as output_file: 
        output_file.write(turtle_data)

    print(f"Résultat sauvegardé dans '{output_file_path}'.")

    # Nettoyage mémoire
    del outputs, inputs, text, turtle_data
    gc.collect()
    torch.cuda.empty_cache()

print("\nProcessing complete for all files and models.")

