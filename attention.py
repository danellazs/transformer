import numpy as np

# softmax standar buat bikin angka jadi probability
def softmax(x, axis=-1):
    # biar aman, kurangi max dulu supaya ga overflow
    x = x - np.max(x, axis=axis, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)

# attention dot product yang diskala
# maksudnya: kita hitung Q*K^T terus dibagi sqrt(d_k)
# biar softmaxnya ga terlalu gede
class ScaledDotProductAttention:
    def __init__(self, d_k):
        self.d_k = d_k  # ukuran dimensi tiap query/key, dipake buat scaling

    def forward(self, Q, K, V, mask=None):
        # hitung score = Q * K^T / sqrt(d_k)
        scores = np.matmul(Q, K.transpose(0,1,3,2)) / np.sqrt(self.d_k)
        
        if mask is not None:  
            # kalo ada mask, yang 0 dibikin -inf biar softmax ga pilih itu
            scores = np.where(mask==0, -1e9, scores)
        
        # softmax biar jadi probabilitas attention
        attn = softmax(scores, axis=-1)
        
        # kalikan sama V biar dapet output weighted sum
        output = np.matmul(attn, V)
        return output, attn

# multi-head attention
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0  # pastiin bisa dibagi rata per head
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # ukuran tiap head

        # projection matrix random, tanpa dilatih
        # fungsinya: ubah Q,K,V ke ruang d_model dulu
        self.Wq = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.Wk = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.Wv = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.Wo = np.random.randn(d_model, d_model) / np.sqrt(d_model)

        # object attention kecil
        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(self, Q, K, V, mask=None):
        B, T, _ = Q.shape  # batch size sama panjang sequence

        # helper: split head
        # dari (B,T,d_model) -> (B,heads,T,d_k)
        def split_heads(x):
            return x.reshape(B, T, self.num_heads, self.d_k).transpose(0,2,1,3)

        # proj Q,K,V dulu baru split ke beberapa head
        Q = split_heads(Q @ self.Wq)
        K = split_heads(K @ self.Wk)
        V = split_heads(V @ self.Wv)

        # hitung attention
        out, attn = self.attention.forward(Q, K, V, mask)

        # simpan attention terakhir untuk visualisasi di test
        self.last_attn_weights = attn

        # gabung head lagi jadi (B,T,d_model)
        out = out.transpose(0,2,1,3).reshape(B, T, self.d_model)

        # projection akhir biar dimensi output sama kaya input
        return out @ self.Wo, attn
