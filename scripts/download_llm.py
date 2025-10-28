from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


MODEL_NAME = 'meta-llama/Llama-3.2-3B' #'Qwen/Qwen2.5-72B-Instruct' #'meta-llama/Llama-3.3-70B-Instruct'
BASE_DIRECTORY = '/home2020/home/icube/fghazoua/TetraProject/models'
BNB_DIRECTORY = f'{BASE_DIRECTORY}/bnb'

# if not os.path.exists(MLX_DIRECTORY):
#     os.makedirs(MLX_DIRECTORY)

model_directory = f'{BASE_DIRECTORY}/{MODEL_NAME}'
bnb_model_directory = f'{BNB_DIRECTORY}/{MODEL_NAME}'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

tokenizer.save_pretrained(save_directory = model_directory)
model.save_pretrained(save_directory = model_directory)

