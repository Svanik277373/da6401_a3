"""
model.py — Transformer Architecture Implementation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
┌─────────────────────────────────────────────────────────────────┐
│ scaled_dot_product_attention(Q, K, V, mask) → (out, weights)   │
│ MultiHeadAttention.forward(q, k, v, mask) → Tensor             │
│ PositionalEncoding.forward(x) → Tensor                         │
│ make_src_mask(src, pad_idx) → BoolTensor                       │
│ make_tgt_mask(tgt, pad_idx) → BoolTensor                       │
│ Transformer.encode(src, src_mask) → Tensor                     │
│ Transformer.decode(memory,src_m,tgt,tgt_m) → Tensor            │
└─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
# STANDALONE ATTENTION FUNCTION
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

    Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    attn_w = F.softmax(scores, dim=-1)
    attn_w = torch.nan_to_num(attn_w, nan=0.0)

    output = torch.matmul(attn_w, V)
    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
# MASK HELPERS
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a padding mask for the encoder (source sequence).
    """
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a combined padding + causal (look-ahead) mask for the decoder.
    """
    batch_size, tgt_len = tgt.size()

    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)

    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool),
        diagonal=1
    ).unsqueeze(0).unsqueeze(0)

    return pad_mask | causal_mask


# ══════════════════════════════════════════════════════════════════════
# MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)
        self.attn_weights = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        Q = self._split_heads(self.W_q(query))
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))

        x, attn_w = scaled_dot_product_attention(Q, K, V, mask)
        self.attn_weights = attn_w

        x = self._merge_heads(x)
        output = self.W_o(x)
        return output


# ══════════════════════════════════════════════════════════════════════
# POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
# FEED-FORWARD NETWORK
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
# ENCODER & DECODER LAYERS
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        self_attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))

        cross_attn_out = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))

        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x


class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape[0])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape[0])

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
# FULL TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.
    """
    def __init__(
        self,
        src_vocab_size: int = 10000,
        tgt_vocab_size: int = 10000,
        d_model: int = 512,
        N: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        inference_mode: bool = True,  # Ensures it's self-contained for autograder
    ) -> None:
        super().__init__()

        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.pos_encoding = PositionalEncoding(d_model, dropout)

        encoder_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(encoder_layer, N)

        decoder_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(decoder_layer, N)

        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        self.d_model = d_model
        self.embed_scale = math.sqrt(d_model)

        self._init_parameters()

        self.src_stoi = {}
        self.tgt_itos = {}

        if inference_mode:
            self._load_for_inference()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _load_for_inference(self):
        """Autograder dependency handling: loads weights and vocabs inside init."""
        import spacy
        
        # NOTE: Provide your google drive ID where the full '.pt' checkpoint is hosted
        file_id = "1DVCPsIZYwQmFUeBNoJS0hYNloD5lcBaZD" 
        ckpt_path = "best_model.pt"

        if file_id != "YOUR_GDRIVE_FILE_ID" and not os.path.exists(ckpt_path):
            import gdown    
            print("Downloading model checkpoint from Google Drive...")
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, ckpt_path, quiet=False)
        elif file_id == "YOUR_GDRIVE_FILE_ID":
            print("WARNING: inference_mode is True but no gdown ID provided. Using randomized weights.")

        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=next(self.parameters()).device)
            config = ckpt.get("model_config", {})

            if config:
                self.src_embedding = nn.Embedding(config["src_vocab_size"], config["d_model"])
                self.tgt_embedding = nn.Embedding(config["tgt_vocab_size"], config["d_model"])
                self.output_projection = nn.Linear(config["d_model"], config["tgt_vocab_size"])

            self.load_state_dict(ckpt["model_state_dict"])

            if "src_vocab_stoi" in ckpt:
                self.src_stoi = ckpt["src_vocab_stoi"]
            if "tgt_vocab_itos" in ckpt:
                self.tgt_itos = ckpt["tgt_vocab_itos"]

        try:
            self.spacy_de = spacy.load("de_core_news_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "de_core_news_sm"])
            self.spacy_de = spacy.load("de_core_news_sm")

        try:
            self.spacy_en = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.spacy_en = spacy.load("en_core_web_sm")

    def infer(self, german_sentence: str) -> str:
        """
        Accepts a german sentence, tokenizes, runs autoregressive decoding, 
        and outputs equivalent english string.
        """
        self.eval()
        device = next(self.parameters()).device

        tokens = [tok.text.lower() for tok in self.spacy_de.tokenizer(german_sentence)]
        
        unk_idx, pad_idx, sos_idx, eos_idx = 0, 1, 2, 3

        indices = [sos_idx] + [self.src_stoi.get(t, unk_idx) for t in tokens] + [eos_idx]
        src = torch.tensor([indices], dtype=torch.long, device=device)
        src_mask = make_src_mask(src, pad_idx=pad_idx).to(device)

        with torch.no_grad():
            memory = self.encode(src, src_mask)
            ys = torch.tensor([[sos_idx]], dtype=torch.long, device=device)

            for _ in range(100):  
                tgt_mask = make_tgt_mask(ys, pad_idx=pad_idx).to(device)
                logits = self.decode(memory, src_mask, ys, tgt_mask)
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ys = torch.cat([ys, next_tok], dim=1)
                
                if next_tok.item() == eos_idx:
                    break

        out_indices = ys[0, 1:].tolist()
        out_tokens = []
        for idx in out_indices:
            if idx == eos_idx:
                break
            out_tokens.append(self.tgt_itos.get(idx, "<unk>"))

        return " ".join(out_tokens)


    # ── AUTOGRADER HOOKS ──────────────────────────────────────────────────

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.pos_encoding(self.src_embedding(src) * self.embed_scale)
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.pos_encoding(self.tgt_embedding(tgt) * self.embed_scale)
        x = self.decoder(x, memory, src_mask, tgt_mask)
        return self.output_projection(x)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)