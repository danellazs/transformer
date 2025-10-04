import numpy as np

# embedding token: tiap token diubah jadi vector d_model
class TokenEmbedding:
    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model
        # random init embeddings, dibagi sqrt(d_model) agar skala wajar
        self.embeddings = np.random.randn(vocab_size, d_model) / np.sqrt(d_model)

    def forward(self, x):
        return self.embeddings[x]

# positional encoding, utk memberi info posisi tiap token
class PositionalEncoding:
    def __init__(self, d_model, max_len=5000):
        self.d_model = d_model
        self.max_len = max_len
        # tempat menyimpan encoding tiap posisi
        self.encoding = np.zeros((max_len, d_model))

        pos = np.arange(0, max_len)[:, None]  # posisi tiap token
        # frekuensi beda-beda buat tiap dimensi embedding
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        self.encoding[:, 0::2] = np.sin(pos * div_term)  # dim genap
        self.encoding[:, 1::2] = np.cos(pos * div_term)  # dim ganjil 

    def forward(self, x):
        seq_len = x.shape[1]  # panjang sequence
        # tambahkan encoding ke embedding
        return x + self.encoding[:seq_len]
