import os
print(os.getcwd())
import torch
import matplotlib.pyplot as plt

# these are the embedding vectors of each word
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your (x^1)
    [0.55, 0.87, 0.66], # journey (x^2)
    [0.57, 0.85, 0.64], # starts (x^3)
    [0.22, 0.58, 0.33], # with (x^4)
    [0.77, 0.25, 0.10], # one (x^5)
    [0.05, 0.80, 0.55]] # step (x^6)
)

print(inputs)



query = inputs[1]

attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)

attn_weights_2_tmp = attn_scores_2/attn_scores_2.sum()

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

plt.imshow([attn_weights_2_naive, attn_weights_2_tmp,attn_weights_2])

# embeddings are in inputs
inputs
# now we calc the context vectors = vector embedding x attn vector

query
attn_weights_2

context_vec_2 = torch.zeros(query.shape)

for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
    print(context_vec_2)


# |> GENERALIZE TO ALL EMBEDDING INPUTS -----------------------------------------------//

# first the similarity scores


















