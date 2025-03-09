from model import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

# Load the entire model
model = torch.load("model.pth", weights_only=False, map_location=torch.device('cpu'))
model.device = device
m = model.to(device)

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
with open('sample.txt', 'a') as f:
    print(decode(m.generate(context, max_new_tokens=1000)[0].tolist()), file=f)