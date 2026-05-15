# DA6401 Assignment 3 — Transformer for Machine Translation

Report Link: https://api.wandb.ai/links/saisvanik2121-iitm-india/dcyj3sx9

Github link: https://github.com/Svanik277373/da6401_a3

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download spaCy language models
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm

# 3. Login to W&B
wandb login
```

## File Structure

```
├── model.py              # Transformer architecture
├── lr_scheduler.py       # Noam LR scheduler
├── dataset.py            # Multi30k loading + vocabulary
├── train.py              # General training pipeline (all autograder functions)
├── requirements.txt
├── README.md
└── runs/
    ├── q2_1_noam_vs_fixed.py        # Section 2.1 — Noam vs fixed LR
    ├── q2_2_scaling_ablation.py     # Section 2.2 — √dk scaling ablation
    ├── q2_3_attention_rollout.py    # Section 2.3 — Attention heatmaps
    ├── q2_4_positional_encoding.py  # Section 2.4 — Sinusoidal vs learned PE
    └── q2_5_label_smoothing.py      # Section 2.5 — Label smoothing ablation
```

## Running

### Baseline training (single run)
```bash
python train.py
# or with custom args:
python train.py --d_model 512 --N 6 --d_ff 2048 --batch_size 64 --num_epochs 20
```

### W&B Report experiments (one script per section)
```bash
python runs/q2_1_noam_vs_fixed.py
python runs/q2_2_scaling_ablation.py
python runs/q2_3_attention_rollout.py
python runs/q2_4_positional_encoding.py
python runs/q2_5_label_smoothing.py
```

## Hardware notes (Ryzen 7 8845HS + RTX 4060 8GB)

| Config | d_model | N | d_ff | batch | VRAM |
|--------|---------|---|------|-------|------|
| Default (fast) | 256 | 3 | 512  | 128   | ~3GB |
| Better BLEU    | 512 | 6 | 2048 | 64    | ~7GB |

## Autograder contract

| Function | File |
|---|---|
| `scaled_dot_product_attention` | model.py |
| `MultiHeadAttention.forward` | model.py |
| `PositionalEncoding.forward` | model.py |
| `make_src_mask` / `make_tgt_mask` | model.py |
| `Transformer.encode` / `.decode` | model.py |
| `NoamScheduler.get_lr` | lr_scheduler.py |
| `greedy_decode` | train.py |
| `evaluate_bleu` | train.py |
| `save_checkpoint` / `load_checkpoint` | train.py |
