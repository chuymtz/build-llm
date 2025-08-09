from src.gpt import GPTModel
from src.tokenizers import tokenizer
from src.utiils import load_config, generate_text_simple
import tiktoken, json
from pprint import pprint
import torch

cfg = load_config("config.json")

model = GPTModel(cfg)   

start_context = "Hello, I am"

encoded = tokenizer.encode(start_context)

encoded_tensor = torch.tensor(encoded).unsqueeze(0)

model.eval()
out = generate_text_simple(
    model=model, 
    idx = encoded_tensor,
    context_size=cfg['context_length'],
    max_new_tokens=40
)
decode_text = tokenizer.decode(out.squeeze().tolist())




