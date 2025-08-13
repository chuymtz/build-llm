import torch
import torch.nn as nn
import json
import os, pprint
import matplotlib.pyplot as plt
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
os.getcwd()
from src.gpt import GPTModel


if __name__ == "__main__":
    with open("config.json", "r") as f:
        GPT_CONFIG_124M = json.load(f)
        pprint.pprint(GPT_CONFIG_124M)

    torch.manual_seed(123)
    model = GPTModel(cfg=GPT_CONFIG_124M)

    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    print(batch.shape)

    out = model(batch)

    print("Input batch:\n", batch)
    print("\nOutput shape:", out.shape)
    print(out)

    total_params = sum(p.numel() for p in model.parameters())
    model.tok_emb.weight.shape
    model.out_head.weight.shape


    sum(p.numel() for p in model.out_head.parameters())
    total_params - sum(p.numel() for p in model.out_head.parameters())