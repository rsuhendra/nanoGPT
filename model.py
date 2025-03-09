import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

class Head(nn.Module):
	def __init__(self, head_size, n_embd, context_size, dropout, bias):
		super().__init__()
		self.key   = nn.Linear(n_embd, head_size, bias=bias)
		self.query = nn.Linear(n_embd, head_size, bias=bias)
		self.value = nn.Linear(n_embd, head_size, bias=bias)
		
		# Registering tril buffer speeds up computation
		self.register_buffer('tril', torch.tril(torch.ones(context_size,context_size))) # tril creation-step (lower triangular)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		# batch, context, n_embd
		B,T,C = x.shape
		k = self.key(x)
		q = self.query(x)
		# compute attention scores (affinities)
		wei = q @ k.transpose(-2,-1) * C**(-0.5) # (B,T,C) @ (B,C,T) ---> (B,T,T)
		# Causality/Autoregressive component
		wei = wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))
		wei = F.softmax(wei, dim=-1)
		wei = self.dropout(wei)
		v = self.value(x)
		out = wei @ v
		return out

class MultiHeadAttention(nn.Module):
	
	def __init__(self, num_heads, head_size, n_embd, context_size, dropout, bias):
		super().__init__()
		# Concatenate num_heads Heads together 
		self.heads = nn.ModuleList([Head(head_size, n_embd, context_size, dropout, bias) for _ in range(num_heads)])
		
		# self.proj  = nn.Linear(n_embd, n_embd, bias=bias)
		self.proj  = nn.Linear(head_size*num_heads, n_embd, bias=bias)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		out = torch.cat([h(x) for h in self.heads], dim=-1)
		out = self.dropout(self.proj(out)) # For ResNet 
		return out
	
class MLP(nn.Module):

	def __init__(self, n_embd, mlp_mult, dropout, bias):
		super().__init__()
		self.layer1 = nn.Linear(n_embd, mlp_mult * n_embd, bias=bias)
		self.gelu = nn.GELU()
		self.layer2  = nn.Linear(mlp_mult * n_embd, n_embd, bias=bias)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		x = self.layer1(x)
		x = self.gelu(x)
		x = self.layer2(x)
		x = self.dropout(x)
		return x

class Block(nn.Module):
	def __init__(self, num_heads, head_size, n_embd, context_size, mlp_mult, dropout, bias):
		super().__init__()
		self.mha = MultiHeadAttention(num_heads, head_size, n_embd, context_size, dropout, bias)
		self.mlp = MLP(n_embd, mlp_mult, dropout, bias)
		self.ln_1 = nn.LayerNorm(n_embd, bias=bias)
		self.ln_2 = nn.LayerNorm(n_embd, bias=bias)

	def forward(self, x):
		# Again, addition here allows for using ResNet 
		x = x + self.mha(self.ln_1(x))
		x = x + self.mlp(self.ln_2(x))
		return x

@dataclass
class GPTConfig:
	# Default config
	vocab_size: int 
	context_size: int = 256
	n_layer: int = 6
	num_heads: int = 6
	head_size: int = 64
	n_embd: int = 384
	mlp_mult: int = 4
	dropout: float = 0.2
	device: str = 'cuda'
	bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config
		self.device = config.device
		
		# Token and position embeddings
		self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
		self.position_embedding_table = nn.Embedding(config.context_size, config.n_embd)

		self.blocks = nn.Sequential(*[Block(config.num_heads, config.head_size, config.n_embd, config.context_size, config.mlp_mult, config.dropout, config.bias) for _ in range(config.n_layer)])
		self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
		self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)
	
	def forward(self, input, targets=None):
		# input and targets are both (Batch, SeqLength) tensor of integers
		B, T = input.shape # Extract the values of Batch & SeqLength

		tok_emb = self.token_embedding_table(input) # (Batch, SeqLength, N_EMBD)  Name the intermediate result and get the intermediate result

		pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (SeqLength, N_EMBD) Get the position embeddings

		x = tok_emb + pos_emb # Add position & token embeddings

		x = self.blocks(x)
		x = self.ln_f(x)
		
		logits = self.lm_head(x) # (Batch, SeqLength, vocab_size) Get logits from intermediate result

		if targets is None:
			loss = None
		else:
			B, T, C = logits.shape
			# Reshape vectors
			logits = logits.view(B*T, C) 
			targets = targets.view(B*T)
			loss = F.cross_entropy(logits, targets)

		return logits, loss
	
	def generate(self, input, max_new_tokens):
		# input is (B, T) array of indices in the current context
		for _ in range(max_new_tokens):
			# crop input to the last context_size tokens
			input_cond = input[:,-self.config.context_size:]
			# get the predictions
			logits, loss = self(input_cond)
			# focus only on the last time step
			logits = logits[:, -1, :] # becomes (B, C)
			# apply softmax to get probabilities
			probs = F.softmax(logits, dim=-1) # (B, C)
			# sample from the distribution
			input_next = torch.multinomial(probs, num_samples=1) # (B, 1)
			# append sampled index to the running sequence
			input = torch.cat((input, input_next), dim=1) # (B, T+1)

		return input

