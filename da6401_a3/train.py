"""
train.py — General Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"
"""

import argparse
import math
import os
import types
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sacrebleu.metrics import BLEU
import wandb

from model import Transformer, make_src_mask, make_tgt_mask
from dataset import (
    build_dataloaders,
    PAD_IDX, SOS_IDX, EOS_IDX,
    Vocabulary,
)
from lr_scheduler import NoamScheduler


# ══════════════════════════════════════════════════════════════════════
# LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            smooth_val = self.smoothing / max(self.vocab_size - 2, 1)
            dist = torch.full_like(logits, smooth_val)
            dist.scatter_(1, target.unsqueeze(1), self.confidence)
            dist[:, self.pad_idx] = 0.0
            pad_mask = target.eq(self.pad_idx)
            dist[pad_mask] = 0.0

        log_probs = torch.log_softmax(logits, dim=-1)
        loss      = -(dist * log_probs).sum(dim=-1)
        n_tokens  = (~pad_mask).sum().clamp(min=1)
        return loss.sum() / n_tokens


# ══════════════════════════════════════════════════════════════════════
# TRAINING / VALIDATION EPOCH
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
    log_wandb: bool = True,
    grad_norm_log: bool = False,
    grad_norm_steps: int = 1000,
) -> Tuple[float, float]:
    """
    Returns:
        (avg_loss, accuracy)
    """
    model.train() if is_train else model.eval()

    total_loss   = 0.0
    total_correct = 0
    total_tokens = 0
    global_step  = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        pbar = tqdm(
            data_iter,
            desc=f"{'Train' if is_train else 'Val  '} Epoch {epoch_num:03d}",
            leave=False,
        )
        for src, tgt in pbar:
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_in  = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            src_mask = make_src_mask(src,    pad_idx=PAD_IDX)
            tgt_mask = make_tgt_mask(tgt_in, pad_idx=PAD_IDX)

            logits = model(src, tgt_in, src_mask, tgt_mask)
            logits_flat = logits.contiguous().view(-1, logits.size(-1))
            tgt_flat    = tgt_out.contiguous().view(-1)

            loss = loss_fn(logits_flat, tgt_flat)
            
            # Accuracy calc
            with torch.no_grad():
                preds = logits_flat.argmax(dim=-1)
                non_pad_mask = tgt_flat.ne(PAD_IDX)
                correct = preds.eq(tgt_flat).logical_and(non_pad_mask).sum().item()
                n_tok = non_pad_mask.sum().item()

            if is_train:
                optimizer.zero_grad()
                loss.backward()

                if grad_norm_log and global_step < grad_norm_steps:
                    qk_norms = {}
                    for name, param in model.named_parameters():
                        if param.grad is not None and (
                            "W_q" in name or "W_k" in name
                        ):
                            key = name.replace(".weight", "").replace(".bias", "")
                            qk_norms[f"grad_norm/{key}"] = param.grad.norm().item()
                    if log_wandb and qk_norms:
                        wandb.log({**qk_norms, "grad_step": global_step})

                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss   += loss.item() * n_tok
            total_correct += correct
            total_tokens += n_tok
            global_step  += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(total_tokens, 1)
    accuracy = total_correct / max(total_tokens, 1)

    if log_wandb:
        prefix = "train" if is_train else "val"
        wandb.log({
            f"{prefix}/loss":       avg_loss,
            f"{prefix}/accuracy":   accuracy,
            f"{prefix}/perplexity": math.exp(min(avg_loss, 100)),
            "epoch": epoch_num,
        })

    return avg_loss, accuracy


# ══════════════════════════════════════════════════════════════════════
# GREEDY DECODING
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        src      = src.to(device)
        src_mask = src_mask.to(device)
        memory   = model.encode(src, src_mask)
        ys       = torch.tensor([[start_symbol]], dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, pad_idx=PAD_IDX).to(device)
            logits   = model.decode(memory, src_mask, ys, tgt_mask)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys       = torch.cat([ys, next_tok], dim=1)
            if next_tok.item() == end_symbol:
                break

    return ys


# ══════════════════════════════════════════════════════════════════════
# BLEU EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab: Vocabulary,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    model.eval()
    bleu_metric = BLEU(effective_order=True)
    hypotheses  = []
    references  = []

    with torch.no_grad():
        for src, tgt in tqdm(test_dataloader, desc="BLEU eval", leave=False):
            src      = src.to(device)
            src_mask = make_src_mask(src, pad_idx=PAD_IDX).to(device)

            ys = greedy_decode(
                model, src, src_mask, max_len,
                start_symbol=SOS_IDX,
                end_symbol=EOS_IDX,
                device=device,
            )

            hyp = []
            for idx in ys[0, 1:].tolist():
                if idx == EOS_IDX:
                    break
                hyp.append(tgt_vocab.lookup_token(idx))

            ref = []
            for idx in tgt[0, 1:].tolist():
                if idx == EOS_IDX:
                    break
                ref.append(tgt_vocab.lookup_token(idx))

            hypotheses.append(" ".join(hyp))
            references.append(" ".join(ref))

    result = bleu_metric.corpus_score(hypotheses, [references])
    return result.score


# ══════════════════════════════════════════════════════════════════════
# CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    ckpt = {
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else {},
        "model_config": {
            "src_vocab_size": model.src_embedding.num_embeddings,
            "tgt_vocab_size": model.tgt_embedding.num_embeddings,
            "d_model":        model.d_model,
            "N":              len(model.encoder.layers),
            "num_heads":      model.encoder.layers[0].self_attn.num_heads,
            "d_ff":           model.encoder.layers[0].ffn.linear1.out_features,
            "dropout":        model.pos_encoding.dropout.p,
        },
    }
    
    # Save vocab mappings if they've been attached to model
    if hasattr(model, "src_vocab_stoi"):
        ckpt["src_vocab_stoi"] = model.src_vocab_stoi
        ckpt["src_vocab_itos"] = model.src_vocab_itos
        ckpt["tgt_vocab_stoi"] = model.tgt_vocab_stoi
        ckpt["tgt_vocab_itos"] = model.tgt_vocab_itos

    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"]


# ══════════════════════════════════════════════════════════════════════
# ATTENTION HEATMAP LOGGING  (Section 2.3)
# ══════════════════════════════════════════════════════════════════════

def log_attention_heatmaps(
    model: Transformer,
    src: torch.Tensor,
    src_vocab: Vocabulary,
    device: str,
) -> None:
    import matplotlib.pyplot as plt

    model.eval()
    with torch.no_grad():
        s    = src[:1].to(device)
        mask = make_src_mask(s, pad_idx=PAD_IDX).to(device)
        model.encode(s, mask)

    attn_weights = model.encoder.layers[-1].self_attn.attn_weights 
    if attn_weights is None:
        return

    tokens = [src_vocab.lookup_token(i.item()) for i in src[0]]
    H      = attn_weights.size(1)
    cols   = min(4, H)
    rows   = math.ceil(H / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten() if H > 1 else [axes]

    for h in range(H):
        w  = attn_weights[0, h].cpu().numpy()
        ax = axes[h]
        im = ax.imshow(w, cmap="viridis", aspect="auto", vmin=0, vmax=1)
        ax.set_title(f"Head {h + 1}", fontsize=9)
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=6)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=6)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for h in range(H, len(axes)):
        axes[h].set_visible(False)

    plt.suptitle("Last Encoder Layer — All Attention Heads", fontsize=12)
    plt.tight_layout()
    wandb.log({"attention/all_heads": wandb.Image(fig)})
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# PREDICTION CONFIDENCE LOGGING  (Section 2.5)
# ══════════════════════════════════════════════════════════════════════

def log_prediction_confidence(
    model: Transformer,
    data_iter,
    device: str,
    n_batches: int = 50,
) -> float:
    model.eval()
    total_conf = 0.0
    total_tok  = 0

    with torch.no_grad():
        for i, (src, tgt) in enumerate(data_iter):
            if i >= n_batches:
                break
            src     = src.to(device)
            tgt     = tgt.to(device)
            tgt_in  = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            src_mask = make_src_mask(src,    pad_idx=PAD_IDX)
            tgt_mask = make_tgt_mask(tgt_in, pad_idx=PAD_IDX)
            logits   = model(src, tgt_in, src_mask, tgt_mask)
            probs    = torch.softmax(logits, dim=-1)

            correct_probs = probs.gather(2, tgt_out.unsqueeze(-1)).squeeze(-1)
            non_pad       = tgt_out.ne(PAD_IDX)
            total_conf   += correct_probs[non_pad].sum().item()
            total_tok    += non_pad.sum().item()

    return total_conf / max(total_tok, 1)


# ══════════════════════════════════════════════════════════════════════
# CORE TRAIN FUNCTION 
# ══════════════════════════════════════════════════════════════════════

def train(
    project:             str   = "da6401-a3",
    run_name:            str   = "baseline",
    d_model:             int   = 256,
    N:                   int   = 3,
    num_heads:           int   = 8,
    d_ff:                int   = 512,
    dropout:             float = 0.1,
    batch_size:          int   = 128,
    min_freq:            int   = 2,
    num_epochs:          int   = 15,
    warmup_steps:        int   = 4000,
    use_noam:            bool  = True,
    fixed_lr:            float = 1e-4,
    label_smoothing:     float = 0.1,
    use_scaling:         bool  = True,   
    use_learned_pos:     bool  = False,  
    log_grad_norms:      bool  = False,  
    log_attn_maps:       bool  = True,   
    log_pred_confidence: bool  = False,  
    checkpoint_dir:      str   = "checkpoints",
) -> float:
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{run_name}]  device={device}")

    # ── W&B ───────────────────────────────────────────────────────────
    wandb.init(
        project=project,
        name=run_name,
        config=dict(
            d_model=d_model, N=N, num_heads=num_heads, d_ff=d_ff,
            dropout=dropout, batch_size=batch_size, num_epochs=num_epochs,
            warmup_steps=warmup_steps, use_noam=use_noam, fixed_lr=fixed_lr,
            label_smoothing=label_smoothing, use_scaling=use_scaling,
            use_learned_pos=use_learned_pos,
        ),
        reinit=True,
    )

    # ── Data ──────────────────────────────────────────────────────────
    print("Loading Multi30k …")
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = build_dataloaders(
        batch_size=batch_size, min_freq=min_freq
    )
    print(f"  src_vocab={len(src_vocab)}  tgt_vocab={len(tgt_vocab)}")

    # ── Model ─────────────────────────────────────────────────────────
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=d_model, N=N, num_heads=num_heads,
        d_ff=d_ff, dropout=dropout,
        inference_mode=False  # Important: Bypass checkpoint fetching during training
    ).to(device)
    
    # Attach to model to serialize inside save_checkpoint hook
    model.src_vocab_stoi = src_vocab.stoi
    model.src_vocab_itos = src_vocab.itos
    model.tgt_vocab_stoi = tgt_vocab.stoi
    model.tgt_vocab_itos = tgt_vocab.itos

    if use_learned_pos:
        class _LearnedPE(nn.Module):
            def __init__(self, d_model, max_len=256, p=0.1):
                super().__init__()
                self.embed   = nn.Embedding(max_len, d_model)
                self.dropout = nn.Dropout(p)
            def forward(self, x):
                pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
                return self.dropout(x + self.embed(pos))
        model.pos_encoding = _LearnedPE(d_model, p=dropout).to(device)

    if not use_scaling:
        def _unscaled_sdpa(Q, K, V, mask=None):
            scores = torch.matmul(Q, K.transpose(-2, -1)) 
            if mask is not None:
                scores = scores.masked_fill(mask, float("-inf"))
            w = F.softmax(scores, dim=-1)
            w = torch.nan_to_num(w, nan=0.0)
            return torch.matmul(w, V), w

        for m in model.modules():
            if m.__class__.__name__ == "MultiHeadAttention":
                def _new_fwd(self, query, key, value, mask=None,
                             _sdpa=_unscaled_sdpa):
                    Q = self._split_heads(self.W_q(query))
                    K = self._split_heads(self.W_k(key))
                    V = self._split_heads(self.W_v(value))
                    x, w = _sdpa(Q, K, V, mask)
                    self.attn_weights = w
                    return self.W_o(self._merge_heads(x))
                m.forward = types.MethodType(_new_fwd, m)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  parameters: {n_params:,}")
    wandb.log({"model/n_params": n_params})

    # ── Optimiser + Scheduler ─────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    if use_noam:
        scheduler = NoamScheduler(optimizer, d_model=d_model,
                                  warmup_steps=warmup_steps)
    else:
        for pg in optimizer.param_groups:
            pg["lr"] = fixed_lr
        scheduler = None

    # ── Loss ──────────────────────────────────────────────────────────
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),
        pad_idx=PAD_IDX,
        smoothing=label_smoothing,
    )

    # ── Training loop ─────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_path     = os.path.join(checkpoint_dir, f"{run_name}_best.pt")
    last_path     = os.path.join(checkpoint_dir, f"{run_name}_last.pt")

    for epoch in range(num_epochs):
        train_loss, train_acc = run_epoch(
            train_loader, model, loss_fn, optimizer, scheduler,
            epoch_num=epoch, is_train=True, device=device, log_wandb=True,
            grad_norm_log=log_grad_norms,
        )
        val_loss, val_acc = run_epoch(
            val_loader, model, loss_fn, None, None,
            epoch_num=epoch, is_train=False, device=device, log_wandb=True,
        )

        if log_pred_confidence:
            conf = log_prediction_confidence(model, val_loader, device)
            wandb.log({"val/pred_confidence": conf, "epoch": epoch})

        current_lr = optimizer.param_groups[0]["lr"]
        wandb.log({"lr": current_lr, "epoch": epoch})
        print(
            f"  Epoch {epoch:03d}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  lr={current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, best_path)
            print(f"    ✓ best checkpoint (val_loss={best_val_loss:.4f})")

        save_checkpoint(model, optimizer, scheduler, epoch, last_path)

    # ── Attention visualisation  ───────────────────────────────────────
    if log_attn_maps:
        sample_src, _ = next(iter(val_loader))
        log_attention_heatmaps(model, sample_src, src_vocab, device)

    # ── Test BLEU ─────────────────────────────────────────────────────
    print("Computing test BLEU …")
    test_bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
    print(f"  Test BLEU: {test_bleu:.2f}")
    wandb.log({"test/bleu": test_bleu})

    wandb.finish()
    return test_bleu


# ══════════════════════════════════════════════════════════════════════
# CLI  
# ══════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DA6401-A3 Transformer trainer")
    p.add_argument("--project",   default="da6401-a3")
    p.add_argument("--run_name",  default="baseline")
    p.add_argument("--d_model",   type=int,   default=256)
    p.add_argument("--N",         type=int,   default=3)
    p.add_argument("--num_heads", type=int,   default=8)
    p.add_argument("--d_ff",      type=int,   default=512)
    p.add_argument("--dropout",   type=float, default=0.1)
    p.add_argument("--batch_size", type=int,  default=128)
    p.add_argument("--min_freq",   type=int,  default=2)
    p.add_argument("--num_epochs",   type=int,   default=15)
    p.add_argument("--warmup_steps", type=int,   default=4000)
    p.add_argument("--use_noam",     type=int,   default=1,
                   help="1=Noam scheduler  0=fixed LR")
    p.add_argument("--fixed_lr",     type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--use_scaling",     type=int, default=1,
                   help="1=with √dk scaling  0=without")
    p.add_argument("--use_learned_pos", type=int, default=0,
                   help="1=learned PE  0=sinusoidal PE")
    p.add_argument("--log_grad_norms",      type=int, default=0)
    p.add_argument("--log_attn_maps",       type=int, default=1)
    p.add_argument("--log_pred_confidence", type=int, default=0)
    p.add_argument("--checkpoint_dir", default="checkpoints")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    train(
        project=args.project,
        run_name=args.run_name,
        d_model=args.d_model,
        N=args.N,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        batch_size=args.batch_size,
        min_freq=args.min_freq,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        use_noam=bool(args.use_noam),
        fixed_lr=args.fixed_lr,
        label_smoothing=args.label_smoothing,
        use_scaling=bool(args.use_scaling),
        use_learned_pos=bool(args.use_learned_pos),
        log_grad_norms=bool(args.log_grad_norms),
        log_attn_maps=bool(args.log_attn_maps),
        log_pred_confidence=bool(args.log_pred_confidence),
        checkpoint_dir=args.checkpoint_dir,
    )