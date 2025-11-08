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

# define a function to transform the dataset entries
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
            row_json = json.loads(row["raw"])
            
            if "messages" not in row_json or not isinstance(row_json["messages"], list):
                raise ValueError("Each 'raw' entry must contain a 'messages' key with a list of messages.")
            
            formatted_messages = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in row_json["messages"]
            ]
            
            row["text"] = tokenizer.apply_chat_template(formatted_messages, tokenize=False)
            return row
        
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in row: {e}")
        except Exception as e:
            raise RuntimeError(f"Error processing row: {e}")

    transformed_dataset = dataset.map(transform_row)
    return transformed_dataset



MODEL_NAME = 'meta-llama/Llama-3.2-3B' #'Qwen/Qwen2.5-72B-Instruct' #'meta-llama/Llama-3.3-70B-Instruct' 
BASE_DIRECTORY = '/home2020/home/icube/fghazoua/TetraProject/models'
BNB_DIRECTORY = f'{BASE_DIRECTORY}/bnb'

model_directory = f'{BASE_DIRECTORY}/{MODEL_NAME}'
bnb_model_directory = f'{BNB_DIRECTORY}/{MODEL_NAME}'

torch_dtype = torch.float16
attn_implementation = "flash_attention_2" 


# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)


# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_directory,
    quantization_config=bnb_config,
    device_map={"": torch.cuda.current_device()}, 
    attn_implementation=attn_implementation
)


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_directory)

if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
    tokenizer.chat_template = None 

model, tokenizer = setup_chat_format(model, tokenizer)

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

dataset = load_from_disk("/.../dataset")
dataset = dataset.shuffle(seed=65).select(range(143))

# Apply the format_chat_template function
dataset = format_chat_template(dataset, tokenizer)

dataset = dataset.train_test_split(test_size=0.1)

# Fine-tuning the model
new_model = "Llama-3.2-3B-populate-ontology"
training_arguments = SFTConfig(
    output_dir=new_model,
    dataset_text_field="text",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    num_train_epochs=1,
    eval_strategy="steps", 
    eval_steps=0.2,
    logging_steps=1,
    warmup_steps=10,
    logging_strategy="steps",
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    group_by_length=True,
    max_seq_length = 2048,
    packing= False
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    tokenizer=tokenizer, 
    args=training_arguments,
)

# Start training
trainer.train()


model.config.use_cache = True

# 
# Save the fine-tuned model
trainer.model.save_pretrained(new_model)
