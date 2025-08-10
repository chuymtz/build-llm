from src.gpt import GPTModel
from src.tokenizers import tokenizer
from src.utiils import load_config, generate_text_simple, text_to_token_ids, token_ids_to_text
import tiktoken, json
from pprint import pprint
import torch

cfg = load_config("config.json")

model = GPTModel(cfg)   

start_context = "Every effort moves you"

token_ids = generate_text_simple(
 model=model,
 idx=text_to_token_ids(start_context, tokenizer),
 max_new_tokens=10,
 context_size=cfg["context_length"]
)

token_ids_to_text(token_ids, tokenizer)

