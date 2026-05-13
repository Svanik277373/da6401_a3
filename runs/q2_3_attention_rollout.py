"""
runs/q2_3_attention_rollout.py
─────────────────────────────────────────────────────────────────────
Section 2.3 — Attention Rollout & Head Specialisation
─────────────────────────────────────────────────────────────────────

Trains the baseline model and then:
  1. Extracts per-head attention weights from the last encoder layer
     for a fixed example sentence and logs a heatmap grid to W&B.
  2. Logs individual head heatmaps as separate W&B images so each
     head can be inspected in isolation.
  3. Computes a simple "specialisation score" (entropy of the
     attention distribution) per head to flag redundant vs focused
     heads.

Usage:
    python runs/q2_3_attention_rollout.py
"""

import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
import wandb

from train import train, save_checkpoint, load_checkpoint
from model import Transformer, make_src_mask
from dataset import build_dataloaders, PAD_IDX, Vocabulary


# ── Step 1: train (or load) a baseline model ─────────────────────────

COMMON = dict(
    project             = "da6401-a3",
    run_name            = "q2_3_attn_rollout",
    d_model             = 256,
    N                   = 3,
    num_heads           = 8,
    d_ff                = 512,
    dropout             = 0.1,
    batch_size          = 128,
    min_freq            = 2,
    num_epochs          = 15,
    warmup_steps        = 4000,
    use_noam            = True,
    label_smoothing     = 0.1,
    use_scaling         = True,
    use_learned_pos     = False,
    log_grad_norms      = False,
    log_attn_maps       = True,   # logs the grid during training
    log_pred_confidence = False,
    checkpoint_dir      = "checkpoints",
)

train(**COMMON)

# ── Step 2: reload best checkpoint and run extended analysis ──────────

device = "cuda" if torch.cuda.is_available() else "cpu"
_, val_loader, _, src_vocab, tgt_vocab = build_dataloaders(batch_size=32, min_freq=2)

ckpt_path = os.path.join("checkpoints", "q2_3_attn_rollout_best.pt")
ckpt      = torch.load(ckpt_path, map_location="cpu")
cfg       = ckpt["model_config"]

model = Transformer(
    src_vocab_size=cfg["src_vocab_size"],
    tgt_vocab_size=cfg["tgt_vocab_size"],
    d_model=cfg["d_model"], N=cfg["N"],
    num_heads=cfg["num_heads"], d_ff=cfg["d_ff"],
    dropout=cfg["dropout"],
).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Pick a fixed sentence from the validation set
sample_src, sample_tgt = next(iter(val_loader))
src      = sample_src[:1].to(device)
src_mask = make_src_mask(src, pad_idx=PAD_IDX).to(device)

with torch.no_grad():
    model.encode(src, src_mask)

attn_weights = model.encoder.layers[-1].self_attn.attn_weights  # [1,H,S,S]
tokens       = [src_vocab.lookup_token(i.item()) for i in src[0]]
H            = attn_weights.size(1)

wandb.init(project="da6401-a3", name="q2_3_head_analysis", reinit=True)

# ── Per-head individual heatmaps ─────────────────────────────────────
for h in range(H):
    w   = attn_weights[0, h].cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 5))
    im  = ax.imshow(w, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_title(f"Head {h + 1}", fontsize=11)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens, fontsize=7)
    plt.colorbar(im)
    plt.tight_layout()
    wandb.log({f"attention/head_{h+1:02d}": wandb.Image(fig)})
    plt.close(fig)

# ── Head entropy (lower = more specialised / focused) ────────────────
import numpy as np

head_entropies = []
for h in range(H):
    w   = attn_weights[0, h].cpu().numpy()   # [S, S]
    # Mean entropy across query positions (ignore all-zero rows)
    row_ent = []
    for row in w:
        row = row + 1e-12
        row = row / row.sum()
        ent = -np.sum(row * np.log(row))
        row_ent.append(ent)
    head_entropies.append(float(np.mean(row_ent)))

# Bar chart of entropy per head
fig, ax = plt.subplots(figsize=(8, 3))
ax.bar(range(1, H + 1), head_entropies, color="steelblue")
ax.set_xlabel("Head index")
ax.set_ylabel("Mean attention entropy")
ax.set_title("Head specialisation (lower entropy = more focused)")
ax.set_xticks(range(1, H + 1))
plt.tight_layout()
wandb.log({"attention/head_entropy": wandb.Image(fig)})
plt.close(fig)

# Log as a table too
entropy_table = wandb.Table(
    columns=["head", "mean_entropy"],
    data=[[h + 1, e] for h, e in enumerate(head_entropies)],
)
wandb.log({"attention/entropy_table": entropy_table})

print("\nHead entropies (lower = more specialised):")
for h, e in enumerate(head_entropies):
    print(f"  Head {h+1:2d}: {e:.4f}")

wandb.finish()
print("\nDone. Individual head heatmaps logged to W&B.")
