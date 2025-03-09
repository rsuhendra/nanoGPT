from model import *

# Training params
batch_size = 32 # how many independent sequences will we process in parallel?
max_iters = 1000
eval_interval = 200
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device is', device)
eval_iters = 200

# Load and clean data 
with open('Shakespeare_input_data.txt', 'r', encoding='utf-8') as f:
    text = f.read()
# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split, model):
    context_size = model.config.context_size
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - context_size, (batch_size,))
    x = torch.stack([data[i:i+context_size] for i in ix])
    y = torch.stack([data[i+1:i+context_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
	out = {}
	model.eval()
	for split in ['train', 'val']:
		losses = torch.zeros(eval_iters)
		for k in range(eval_iters):
			X, Y = get_batch(split, model)
			logits, loss = model(X, Y)
			losses[k] = loss.item()
		out[split] = losses.mean()
	model.train()
	return out

cfg = GPTConfig(vocab_size = vocab_size, device = device)
model = GPT(cfg) # Remove vocab_size when evoking the constructor
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

	# every once in a while evaluate the loss on train and val sets
	if (iter % eval_interval == 0) or (iter == (max_iters - 1)):
		losses = estimate_loss(model)
		print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

	# sample a batch of data
	xb, yb = get_batch('train', model)

	# evaluate the loss
	logits, loss = model(xb, yb)
	optimizer.zero_grad(set_to_none=True)
	loss.backward()
	optimizer.step()


num_params = sum(p.numel() for p in model.parameters())
print(f'Number of model parameters: {num_params}')
# Save the entire model
torch.save(model, "model.pth")