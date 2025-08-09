import torch
import json
from pprint import pprint

def load_config(config_path):
    with open(config_path, "r") as f:
        GPT_CONFIG_124M = json.load(f)
        pprint(GPT_CONFIG_124M)
    return GPT_CONFIG_124M

def generate_text_simple(model, idx, max_new_tokens, context_size):

    for _ in range(max_new_tokens):
        # torch.Size([1, 4])
        # idx_cond takes a slice of idx tensor, keeping all rows (:) but only the last context_size 
        # columns (-context_size:)
        # This creates a sliding window over the input sequence of size context_size
        idx_cond = idx[:, -context_size:]
        # above: want to grab as many  "words" as the context size allows starting from the end of the doc
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((
              idx, idx_next
              ), dim=1)
    return idx

