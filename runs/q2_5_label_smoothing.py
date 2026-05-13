"""
runs/q2_5_label_smoothing.py
─────────────────────────────────────────────────────────────────────
Section 2.5 — Decoder Sensitivity: Label Smoothing
─────────────────────────────────────────────────────────────────────

Trains two models:
  Run A — Label smoothing ε = 0.1  (paper default)
  Run B — Label smoothing ε = 0.0  (standard cross-entropy)

Both runs log:
  • train/loss and val/loss curves
  • val/pred_confidence  — mean softmax probability of the correct
    token on the validation set; this directly shows over-confidence
    in Run B vs calibrated confidence in Run A.
  • train/perplexity and val/perplexity

Usage:
    python runs/q2_5_label_smoothing.py
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
    use_scaling         = True,
    use_learned_pos     = False,
    log_grad_norms      = False,
    log_attn_maps       = False,
    log_pred_confidence = True,    # ← key: logs softmax confidence each epoch
    checkpoint_dir      = "checkpoints",
)

# ── Run A: label smoothing ε = 0.1 ───────────────────────────────────
print("=" * 60)
print("Run A: Label smoothing ε = 0.1")
print("=" * 60)
bleu_smooth = train(
    **COMMON,
    run_name        = "q2_5_smooth_0.1",
    label_smoothing = 0.1,
)

# ── Run B: no label smoothing (standard CE) ───────────────────────────
print("=" * 60)
print("Run B: Label smoothing ε = 0.0  (plain cross-entropy)")
print("=" * 60)
bleu_plain = train(
    **COMMON,
    run_name        = "q2_5_smooth_0.0",
    label_smoothing = 0.0,
)

print("\n── Summary ──────────────────────────────")
print(f"  ε=0.1 (smoothed) BLEU : {bleu_smooth:.2f}")
print(f"  ε=0.0 (plain CE) BLEU : {bleu_plain:.2f}")
print(
    "Compare val/pred_confidence curves in W&B:\n"
    "  Run B (plain CE) should show higher confidence but worse generalisation.\n"
    "  Run A (smoothed) should show lower confidence but better BLEU."
)
