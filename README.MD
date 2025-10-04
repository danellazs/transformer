# Implementasi Transformer Decoder dari Nol (NumPy)

## Deskripsi Singkat
Proyek ini adalah implementasi **decoder-only Transformer (GPT-style)** menggunakan **NumPy**.  
Fokusnya pada alur **forward pass**, mulai dari embedding hingga menghasilkan distribusi probabilitas token berikutnya (softmax).  

Arsitektur mencakup:
- **Token Embedding**: ubah token integer jadi vektor representasi.
- **Positional Encoding**: info posisi token (sinusoidal).
- **Scaled Dot-Product Attention & Multi-Head Attention**: relevansi antar token.
- **Feed-Forward Network (FFN)**: dua lapis, aktivasi ReLU.
- **Residual + LayerNorm (pre-norm)**: stabilisasi forward pass.
- **Causal Masking**: mencegah token lihat masa depan.
- **Output Layer**: proyeksi ke vocab + softmax distribusi.

## Struktur Direktori

```

C:\Users\user\nlp
│
├─ attention.py       # Scaled Dot-Product & Multi-Head Attention
├─ embed.py           # TokenEmbedding & PositionalEncoding
├─ layers.py          # TransformerBlock, FeedForward, LayerNorm
├─ transformer.py     # TransformerDecoder (gabungan semua komponen)
├─ main.py            # Contoh forward pass, softmax, mask, attention
└─ test.py            # Uji sederhana: shape, softmax, mask, attention

````

## Cara Menjalankan

1. Aktifkan virtual environment:
```powershell
.\.venv\Scripts\activate
````

2. Install dependensi:

```powershell
python -m pip install numpy
```

3. Jalankan program:

```powershell
python main.py
```

## Output yang Dihasilkan

* **Logits shape**: `[batch, seq_len, vocab_size]`
* **Softmax token terakhir**: distribusi probabilitas token berikutnya
* **Causal mask**: token mana yang boleh diakses
* **Attention map**: bobot perhatian antar token (numerik di terminal)

## Input & Output

* **Input**: sequence token integer, contoh `[1, 5, 7, 10]`.
* **Output**:

  * `logits`: tensor `[batch, seq_len, vocab_size]`
  * `softmax`: distribusi token berikutnya (posisi terakhir)
  * `attention`: bobot perhatian antar token (head/layer tertentu)
  * `causal_mask`: matriks 0/1 menandakan token yang boleh dilihat

## Catatan

* Hanya menggunakan **NumPy**, tidak ada deep learning library.
* Semua komponen **modular** di file terpisah.
* Visualisasi attention bisa dicetak numerik di terminal (tanpa matplotlib).
* Forward pass, softmax, dan mask dapat langsung dicek pakai `main.py`.
