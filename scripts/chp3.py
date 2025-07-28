import torch
import matplotlib.pyplot as plt

inputs = torch.tensor(
[[0.43, 0.15, 0.89], # Your (x^1)
[0.55, 0.87, 0.66], # journey (x^2)
[0.57, 0.85, 0.64], # starts (x^3)
[0.22, 0.58, 0.33], # with (x^4)
[0.77, 0.25, 0.10], # one (x^5)
[0.05, 0.80, 0.55]] # step (x^6)
)

# make interesting plot of the input data
plt.scatter(inputs[:, 0], inputs[:, 1], c=inputs[:, 2], cmap='viridis')
plt.colorbar(label='x^3')
plt.xlabel('x^1')

print(f"First word with vector x_0 corresponds to {inputs[0]=} which is the transformed as token embeddings")

# what we now want is to calculate the context vectors z_i for each x_i. Think of it like 
# a richer embedding.

# For instance, x_2 (second word) will generate z_2 which has information about x_1, x_2, and x_3
# and so on. The context vector z_i is a weighted sum of the input vectors x_j, where the weights






