"""
runs/q2_2_scaling_ablation.py
─────────────────────────────────────────────────────────────────────
Section 2.2 — Ablation: The Scaling Factor 1/√dk
─────────────────────────────────────────────────────────────────────

Trains the same model twice:
  Run A — Standard attention with 1/√dk scaling
  Run B — Attention WITHOUT scaling (raw dot products)

Both runs log Q/K gradient norms for the first 1 000 optimisation
steps so you can see the vanishing-gradient effect in the unscaled
version (norms collapse) vs the scaled one (norms stay healthy).

Usage:
    python runs/q2_2_scaling_ablation.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import train

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
    use_learned_pos     = False,
    log_grad_norms      = True,    # ← key: logs Q/K grad norms first 1k steps
    log_attn_maps       = False,
    log_pred_confidence = False,
    checkpoint_dir      = "checkpoints",
)

# ── Run A: with √dk scaling ───────────────────────────────────────────
print("=" * 60)
print("Run A: With √dk scaling (standard)")
print("=" * 60)
train(
    **COMMON,
    run_name    = "q2_2_with_scaling",
    use_scaling = True,
)

# ── Run B: without √dk scaling ───────────────────────────────────────
print("=" * 60)
print("Run B: Without √dk scaling")
print("=" * 60)
train(
    **COMMON,
    run_name    = "q2_2_no_scaling",
    use_scaling = False,
)

print("\nDone. Compare grad_norm/W_q and grad_norm/W_k curves in W&B.")
