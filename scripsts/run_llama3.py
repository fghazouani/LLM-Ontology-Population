import torch
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
#import os, torch, wandb
#from datasets import load_dataset
from trl import SFTTrainer, setup_chat_format

MODEL_NAME = 'meta-llama/Llama-3.3-70B-Instruct' #'meta-llama/Llama-3.2-3B-Instruct'
BASE_DIRECTORY = '/home2020/home/icube/fghazoua/TetraProject/models'
BNB_DIRECTORY = f'{BASE_DIRECTORY}/bnb'

model_directory = f'{BASE_DIRECTORY}/{MODEL_NAME}'
bnb_model_directory = f'{BNB_DIRECTORY}/{MODEL_NAME}'

torch_dtype = torch.float16
attn_implementation = "eager"

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
    device_map="auto",
    attn_implementation=attn_implementation
)


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_directory)

if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
    tokenizer.chat_template = None  # Reset the chat template

model, tokenizer = setup_chat_format(model, tokenizer)

messages = [
    {
        "role": "user",
        "content": "Give me a recipe for delicious chocolate chip cookies"
    }
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, 
                                       add_generation_prompt=True)

inputs = tokenizer(prompt, return_tensors='pt', padding=True, 
                   truncation=True).to("cuda")

outputs = model.generate(**inputs, max_length=150, 
                         num_return_sequences=1)

text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(text)

