import wandb
import torch
import json
import os
import time
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)

#from datasets import load_dataset
from trl import SFTTrainer, setup_chat_format, SFTConfig
from datasets import Dataset, DatasetDict, load_from_disk




def format_chat_template(dataset, tokenizer):
    """
    Transforms each 'raw' entry in the dataset into a formatted chat template.
    
    Args:
        dataset: The dataset object loaded from disk, containing a 'raw' field.
        tokenizer: A tokenizer object with the `apply_chat_template` method.
    
    Returns:
        A new dataset with an additional 'text' field.
    """

    def transform_row(row):
        try:
            # Load the JSON string from the 'raw' field
            row_json = json.loads(row["raw"])
            
            # Ensure the 'messages' field exists
            if "messages" not in row_json or not isinstance(row_json["messages"], list):
                raise ValueError("Each 'raw' entry must contain a 'messages' key with a list of messages.")
            
            # Reformat the 'messages' into the desired chat template format
            formatted_messages = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in row_json["messages"]
            ]
            
            # Apply the chat template using the tokenizer
            row["text"] = tokenizer.apply_chat_template(formatted_messages, tokenize=False)
            return row
        
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in row: {e}")
        except Exception as e:
            raise RuntimeError(f"Error processing row: {e}")

    # Apply the transformation to the dataset
    transformed_dataset = dataset.map(transform_row)
    return transformed_dataset



MODEL_NAME = 'meta-llama/Llama-3.2-3B' #'Qwen/Qwen2.5-72B-Instruct' #'meta-llama/Llama-3.3-70B-Instruct' #'meta-llama/Llama-3.2-3B-Instruct'
BASE_DIRECTORY = '/home2020/home/icube/fghazoua/TetraProject/models'
BNB_DIRECTORY = f'{BASE_DIRECTORY}/bnb'

model_directory = f'{BASE_DIRECTORY}/{MODEL_NAME}'
bnb_model_directory = f'{BNB_DIRECTORY}/{MODEL_NAME}'

torch_dtype = torch.float16
# attn_implementation = "eager" 
attn_implementation = "flash_attention_2" # need pip install -qqq flash-attn


# initialize wandb
wandb.login(key=os.getenv("WANDB_API_KEY"))
run = wandb.init(
    mode="offline",
    project='Fine-tune Llama-3.2-3B on Ontology Populatin', 
    job_type="training", 
    anonymous="allow",
    config={                             # Configuration options
        "model": "Llama-3.2-3B",
        "task": "ontology_population",
        "gpu_count": os.getenv("SLURM_GPUS_ON_NODE", "unknown"),  # GPU info from SLURM
    }
    
)

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)

print("Process started....")
start_time = time.time()
# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_directory,
    quantization_config=bnb_config,
    # device_map="auto", # Gros modèle (70B)
    device_map={"": torch.cuda.current_device()}, # Modèle LLM petit (3B)
    attn_implementation=attn_implementation
)


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_directory)

if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
    tokenizer.chat_template = None  # Reset the chat template

model, tokenizer = setup_chat_format(model, tokenizer)

# Adding the adapter to the layer:
# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)
model = get_peft_model(model, peft_config)


# Loading the dataset

dataset = load_from_disk("/home2020/home/icube/fghazoua/TetraProject/scripts/dataset")
dataset = dataset.shuffle(seed=65).select(range(143))

# Apply the format_chat_template function
dataset = format_chat_template(dataset, tokenizer)

dataset = dataset.train_test_split(test_size=0.1)

# Complaining and training the model
# new_model = "llama-3-70b-populate-ontology"
new_model = "Llama-3.2-3B-populate-ontology"
training_arguments = SFTConfig(
    output_dir=new_model,
    dataset_text_field="text",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    num_train_epochs=1,
    eval_strategy="steps", #Use `eval_strategy` instead
    eval_steps=0.2,
    logging_steps=1,
    warmup_steps=10,
    logging_strategy="steps",
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    group_by_length=True,
    max_seq_length = 8192,#2048,
    packing= False,
    report_to="wandb"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    tokenizer=tokenizer, #Use `processing_class` instead
    args=training_arguments,
)

trainer.train()


# Model evaluation
wandb.finish()
model.config.use_cache = True

# testing the new model

messages = [
    {
        "role": "system",
        "content": "Translate the user text into an TTL graph based on the TetraOnto ontology."
    },
    {
        "role": "user",
        "content": "A restoration project is underway to create a series of pools in the Seine river in Paris, France. The project is managed by the Seine-Normandie Water Agency and the main contractor is Suez. The pools are designed to provide resting areas for migratory fish. The project started in 2018 and is expected to be completed by 2023, with a total cost of 250,000 €."
    }
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, 
                                       add_generation_prompt=True)

inputs = tokenizer(prompt, return_tensors='pt', padding=True, 
                   truncation=True).to("cuda")

outputs = model.generate(**inputs, max_length=8192, 
                         num_return_sequences=1)
# max_length=2048,
text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(text.split("assistant")[1])

end_time = time.time()
execution_time = end_time - start_time
minutes, seconds = divmod(execution_time, 60)
formatted_time = f"{int(minutes)} minutes and {seconds:.2f} seconds"
print(f"Execution time for model '{MODEL_NAME}' is: {formatted_time}")

# Save the fine-tuned model
trainer.model.save_pretrained(new_model)
