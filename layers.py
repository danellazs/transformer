import numpy as np
from attention import MultiHeadAttention

# normalisasi layer agar training lebih stabil
class LayerNorm:
    def __init__(self, d_model, eps=1e-6):
        # gamma dan beta buat skala + shift
        self.gamma = np.ones((1,1,d_model))
        self.beta = np.zeros((1,1,d_model))
        self.eps = eps

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)  # rata-rata tiap token
        var = np.var(x, axis=-1, keepdims=True)    # variansi tiap token
        # normalize + scale + shift
        return self.gamma * (x - mean) / np.sqrt(var + self.eps) + self.beta

# feed-forward layer sederhana (2 linear + ReLU)
class FeedForward:
    def __init__(self, d_model, d_ff):
        # projection ke dimensi lebih besar
        self.W1 = np.random.randn(d_model, d_ff) / np.sqrt(d_model)
        self.b1 = np.zeros((d_ff,))
        # projection balik ke d_model
        self.W2 = np.random.randn(d_ff, d_model) / np.sqrt(d_ff)
        self.b2 = np.zeros((d_model,))

    def forward(self, x):
        # linear + ReLU + linear
        return np.maximum(0, x @ self.W1 + self.b1) @ self.W2 + self.b2

# 1 block transformer: attention + feedforward + 2 layernorm + residual
class TransformerBlock:
    def __init__(self, d_model, num_heads, d_ff):
        self.ln1 = LayerNorm(d_model)               # layer norm sebelum attention
        self.ln2 = LayerNorm(d_model)               # layer norm sebelum feedforward
        self.mha = MultiHeadAttention(d_model, num_heads)  # multi-head attention
        self.ffn = FeedForward(d_model, d_ff)      # feedforward

    def forward(self, x, mask=None):
        # attention
        attn_out, _ = self.mha.forward(
            self.ln1.forward(x), 
            self.ln1.forward(x), 
            self.ln1.forward(x), 
            mask
        )
        x = x + attn_out  # residual

        # feed-forward
        ffn_out = self.ffn.forward(self.ln2.forward(x))
        x = x + ffn_out   # residual
        return x
