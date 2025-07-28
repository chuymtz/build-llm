import os
os.getcwd()
from importlib.metadata import version
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
print("tiktoken version:", version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")
                                  
text = (
"Hello, do you like tea? <|endoftext|> In the sunlit terraces"
"of someunknownPlace."
)    

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

strings = tokenizer.decode(integers)

strings

# |> SLIDE WINDOWS DATA -------------------------------------------------------

with open("data/the-verdict.txt","r",encoding='utf-8') as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

enc_sample = enc_text[:50]

context_size=4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]

x
y

for i in range(1, context_size+1):
    print(i)

class GTPDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        super().__init__()
        self.input_ids = []
        self.target_ids = []
        
        token_ids = tokenizer.encode(txt)
        
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
            
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]

def create_dataloader_v1(
    txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True,
    num_workers=0
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GTPDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, 
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=True)


data_iterator = iter(dataloader)

next(data_iterator)


