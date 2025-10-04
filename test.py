import numpy as np
from transformer import TransformerDecoder

# dummy vocab sama sequence contoh
vocab_size = 50
seq = np.array([[1, 5, 7, 10]])  # batch=1, seq_len=4

model = TransformerDecoder(vocab_size, d_model=32, num_heads=2, d_ff=64, num_layers=2, max_len=20)
logits, _ = model.forward(seq)  # shape: [batch, seq_len, vocab_size]
print(" Logits shape:", logits.shape)  # cek ukuran tensor

# ambil prediksi softmax untuk token terakhir
last_logits = logits[0, -1]

# softmax manual
exp_logits = np.exp(last_logits)
probs = exp_logits / np.sum(exp_logits)

print("\n Softmax distribusi token terakhir:")
print(probs)
print("sum probabilitas (harus 1):", probs.sum())

# cek mask causal
mask = model.causal_mask(seq.shape[1])[0,0]
print("\n Causal mask (1=boleh diakses, 0=tidak boleh):")
print(mask)

# ambil attention dari layer 0, head 0
attn = model.layers[0].mha.last_attn_weights
attn_map = attn[0, 0]  # batch 0, head 0

# tampilkan attention di terminal pake ASCII
def print_attention_numeric(attn_map):
    seq_len = attn_map.shape[0]
    # header kolom (key positions)
    header = "   " + "  ".join([f"{i:>5}" for i in range(seq_len)])
    print(header)
    
    for i, row in enumerate(attn_map):
        line = f"{i:>5}  "  # label baris (query)
        for val in row:
            line += f"{val:0.2f}  "
        print(line)

# pakai seperti ini
attn = model.layers[0].mha.last_attn_weights
attn_map = attn[0, 0]  # batch 0, head 0

print("\n Attention Head 0, Layer 0 (numeric):")
print_attention_numeric(attn_map)


print("\n Semua bukti uji: forward pass, softmax, causal mask, dan attention.")
