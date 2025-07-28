 # build-llm
 
 This repository contains a simple implementation of core components for a GPT-style language model using PyTorch. It is designed for educational and experimental purposes, allowing you to explore transformer blocks, attention mechanisms, and feedforward layers.
 
 ## Features
 
 - **LayerNorm**: Custom layer normalization module.
 - **GELU Activation**: Implementation of the Gaussian Error Linear Unit (GELU) activation function.
 - **FeedForward**: Standard transformer feedforward block.
 - **CausalAttention & MultiHeadAttention**: Implements causal (masked) attention and multi-head attention.
 - **TransformerBlock**: Placeholder for transformer block logic.
 - **DummyGPTModel**: Assembles embeddings, transformer blocks, and output head for a minimal GPT-like model.
 
 ## Project Structure
 
 - `main.py`: Main code for model components and example usage.
 - `config.json`: Model configuration parameters (e.g., embedding dimension, vocab size).
 - `requirements.txt`: Python dependencies.
 - `data/`: Example data files.
 - `src/`: Additional modules (e.g., tokenizers).
 - `scripts/`: Utility scripts for data loading, BPE, and self-attention experiments.
 
 ## Example Usage
 
 ```python
 import torch
 from main import DummyGPTModel
 
 # Load config
 with open("config.json", "r") as f:
     cfg = json.load(f)
 
 # Initialize model
 model = DummyGPTModel(cfg)
 
 # Dummy input (batch_size=2, seq_len=3)
 dummy_input = torch.tensor([[1, 5, 2], [4, 3, 0]])
 output = model.tok_emb(dummy_input)
 print(output.shape)  # (2, 3, emb_dim)
 ```
 
 ## Requirements
 
 - Python 3.8+
 - PyTorch
 - matplotlib
 
 Install dependencies:
 
 ```bash
 pip install -r requirements.txt
 ```
 
 ## License
 
 MIT License
