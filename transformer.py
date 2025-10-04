import numpy as np
from embed import TokenEmbedding, PositionalEncoding
from layers import TransformerBlock

# decoder transformer sederhana
class TransformerDecoder:
    def __init__(self, vocab_size, d_model=128, num_heads=4, d_ff=512, num_layers=2, max_len=100):
        # embedding token
        self.embedding = TokenEmbedding(vocab_size, d_model)
        # positional encoding agar model tau posisi token
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        # tumpukan transformer block
        self.layers = [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        # projection terakhir ke vocab size (utk prediksi token)
        self.Wo = np.random.randn(d_model, vocab_size) / np.sqrt(d_model)

    # membuat mask lower triangular, agar token cuma lihat token sebelumnya (causal)
    def causal_mask(self, seq_len):
        return np.tril(np.ones((1,1,seq_len,seq_len)))

    def forward(self, x):
        # token -> embedding
        x = self.embedding.forward(x)
        x = self.positional_encoding.forward(x)
        mask = self.causal_mask(x.shape[1])

        # melewati semua layer
        for layer in self.layers:
            x = layer.forward(x, mask)

        # projection ke vocab size -> logits prediksi token
        logits = x @ self.Wo
        return logits, None  # None karena belum implement attention return
