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

with open("data/the-verdict.txt", "r", encoding="utf-8") as file:
    text_data = file.read()


total_characters = len(text_data)

total_tokens = len(tokenizer.encode(text_data))
total_tokens


# |> DATA PREP ----------------------------------------------------------//

train_ratio = .9
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]




# |> LOSS ----------------------------------------------------------//

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss

