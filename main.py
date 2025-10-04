import numpy as np
from transformer import TransformerDecoder

# dummy data, 1 sequence panjang 4 token
vocab_size = 50
seq = np.array([[1, 5, 7, 10]])  # batch=1, seq_len=4

model = TransformerDecoder(vocab_size)
logits = model.forward(seq)

# shape: (batch, seq_len, vocab_size)
print("Logits shape:", logits.shape)  

last_logits = logits[0, -1]

probs = np.exp(last_logits) / np.sum(np.exp(last_logits))  

print("Last token distribution (softmax):")  
print(probs)  # tiap index = probabilitas token di vocab
