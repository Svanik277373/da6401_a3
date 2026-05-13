"""
runs/q2_4_positional_encoding.py
─────────────────────────────────────────────────────────────────────
Section 2.4 — Positional Encoding vs Learned Embeddings
─────────────────────────────────────────────────────────────────────

Trains two models:
  Run A — Sinusoidal positional encoding (original paper §3.5)
  Run B — Learned positional embeddings (torch.nn.Embedding)

Both runs log validation BLEU at the end of each epoch so the
learning curves can be directly compared.

Also logs a visualisation of the sinusoidal PE matrix to illustrate
the periodic structure that enables length extrapolation.

Usage:
    python runs/q2_4_positional_encoding.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import matplotlib.pyplot as plt
import wandb

from train import train, evaluate_bleu
from dataset import build_dataloaders


# ── Helper: visualise the sinusoidal PE matrix ───────────────────────

def log_sinusoidal_pe_visual(d_model: int = 256, max_len: int = 100):
    """Log a heatmap of the first max_len × d_model PE values."""
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len).unsqueeze(1).float()
    div = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(pe.numpy().T, cmap="RdBu", aspect="auto",
                   origin="lower", vmin=-1, vmax=1)
    ax.set_xlabel("Position")
    ax.set_ylabel("Embedding dimension")
    ax.set_title("Sinusoidal Positional Encoding")
    plt.colorbar(im)
    plt.tight_layout()
    wandb.init(project="da6401-a3", name="q2_4_pe_visual", reinit=True)
    wandb.log({"pe/sinusoidal_matrix": wandb.Image(fig)})
    plt.close(fig)
    wandb.finish()
    print("PE visualisation logged.")


log_sinusoidal_pe_visual()

# ── Shared hyperparameters ────────────────────────────────────────────
COMMON = dict(
    project             = "da6401-a3",
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
    log_grad_norms      = False,
    log_attn_maps       = False,
    log_pred_confidence = False,
    checkpoint_dir      = "checkpoints",
)

# ── Run A: sinusoidal PE ──────────────────────────────────────────────
print("=" * 60)
print("Run A: Sinusoidal Positional Encoding")
print("=" * 60)
bleu_sin = train(
    **COMMON,
    run_name        = "q2_4_sinusoidal_pe",
    use_learned_pos = False,
)

# ── Run B: learned PE ────────────────────────────────────────────────
print("=" * 60)
print("Run B: Learned Positional Embeddings")
print("=" * 60)
bleu_learned = train(
    **COMMON,
    run_name        = "q2_4_learned_pe",
    use_learned_pos = True,
)

print("\n── Summary ──────────────────────────────")
print(f"  Sinusoidal PE BLEU : {bleu_sin:.2f}")
print(f"  Learned PE BLEU    : {bleu_learned:.2f}")
print("Compare validation BLEU curves in W&B under the da6401-a3 project.")
