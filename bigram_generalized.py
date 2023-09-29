import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = 32
block_size = 20
max_epochs = 10000
torch.manual_seed(103)
n_embed = 64

filename = 'datasets/vietnamese/1984.txt'
with open(filename, 'r', encoding='utf-8') as file:
    text = file.read()

stoi = {ch: i for i, ch in enumerate(sorted(set(text)))}
itos = {i: ch for i, ch in enumerate(sorted(set(text)))}
vocab_size = len(stoi)

encode = lambda s: [stoi[ch] for ch in s]
decode = lambda x: ''.join([itos[i] for i in x])

data = torch.tensor(encode(text), dtype=torch.long)

n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    idx = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([train_data[i:i+block_size] for i in idx])
    y = torch.stack([train_data[i+1:i+block_size+1] for i in idx])
    return x, y

def estimate_loss(model, split):
    model.eval()
    with torch.no_grad():
        x, y = get_batch(split)
        _, loss = model(x, y)
    model.train()
    return loss.item()

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.embedding(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T)) # (T, C)
        x = tok_emb + pos_emb
        logits = self.lm_head(x)
        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # get predictions
            logits, loss = self(idx)
            # focus on the last time steps
            logits = logits[:, -1, :]
            # sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            # update idx
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Create model and training
m = BigramLanguageModel(vocab_size)
optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)

for steps in range(max_epochs):
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if steps % 100 == 0:
        print(f'Step: {steps}, Eval Loss: {estimate_loss(m, "val"):.4f}')

logits, loss = m(xb, yb)
idx = torch.zeros((1, 1), dtype=torch.long)
#m.generate(idx, max_new_tokens=7)[0].tolist()
print(decode(m.generate(idx, max_new_tokens=20)[0].tolist()))