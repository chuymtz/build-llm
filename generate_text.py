from main import GPTModel, GPT_CONFIG_124M
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
import torch

def generate_text_simple(model, idx, max_new_tokens, context_size):

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size]

        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((
              idx, idx_next
              ), dim=1)
    return idx

model = GPTModel(GPT_CONFIG_124M)   

start_context = "Hello, I am"

encoded = tokenizer.encode(start_context)

encoded_tensor = torch.tensor(encoded).unsqueeze(0)

model.eval()
out = generate_text_simple(
    model=model, 
    idx = encoded_tensor,
    context_size=GPT_CONFIG_124M['context_length'],
    max_new_tokens=6
)


