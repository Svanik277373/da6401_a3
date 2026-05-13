"""
runs/q2_1_noam_vs_fixed.py
─────────────────────────────────────────────────────────────────────
Section 2.1 — The Necessity of the Noam Scheduler
─────────────────────────────────────────────────────────────────────

Trains the same model twice:
  Run A — Noam scheduler (linear warmup + inverse-sqrt decay)
  Run B — Constant learning rate of 1e-4 (no warmup)

Both runs log to the same W&B project so curves can be overlaid.

Usage:
    python runs/q2_1_noam_vs_fixed.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import train

# ── Shared hyperparameters ────────────────────────────────────────────
COMMON = dict(
    project        = "da6401-a3",
    d_model        = 256,
    N              = 3,
    num_heads      = 8,
    d_ff           = 512,
    dropout        = 0.1,
    batch_size     = 128,
    min_freq       = 2,
    num_epochs     = 15,
    warmup_steps   = 4000,
    label_smoothing= 0.1,
    use_scaling    = True,
    use_learned_pos= False,
    log_grad_norms      = False,
    log_attn_maps       = False,   
    log_pred_confidence = False,
    checkpoint_dir = "checkpoints",
)

# ── Run A: Noam scheduler ────────────────────────────────────────────
print("=" * 60)
print("Run A: Noam Scheduler")
print("=" * 60)
train(
    **COMMON,
    run_name  = "q2_1_noam",
    use_noam  = True,
)

# ── Run B: Fixed learning rate ───────────────────────────────────────
print("=" * 60)
print("Run B: Fixed LR = 1e-4")
print("=" * 60)
train(
    **COMMON,
    run_name  = "q2_1_fixed_lr",
    use_noam  = False,
    fixed_lr  = 1e-4,
)

print("\nDone. Open your W&B project and overlay the two runs to compare.")