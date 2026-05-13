"""
dataset.py — Multi30k Dataset Loading and Preprocessing
DA6401 Assignment 3

Loads German→English Multi30k from local JSONL files, tokenizes with spaCy,
builds vocabularies (with <unk>, <pad>, <sos>, <eos>), and provides
a collate function for DataLoader batching.
"""

import os
import json
from collections import Counter
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import spacy


# ──────────────────────────────────────────────────────────────────────
# Special token constants
# ──────────────────────────────────────────────────────────────────────
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

PAD_IDX = 1
UNK_IDX = 0
SOS_IDX = 2
EOS_IDX = 3


# ──────────────────────────────────────────────────────────────────────
# Vocabulary
# ──────────────────────────────────────────────────────────────────────

class Vocabulary:
    """Simple vocabulary with stoi / itos mappings."""

    def __init__(self):
        self.stoi: Dict[str, int] = {
            UNK_TOKEN: UNK_IDX,
            PAD_TOKEN: PAD_IDX,
            SOS_TOKEN: SOS_IDX,
            EOS_TOKEN: EOS_IDX,
        }
        self.itos: Dict[int, str] = {v: k for k, v in self.stoi.items()}

    def build_from_counter(self, counter: Counter, min_freq: int = 2):
        for token, freq in counter.most_common():
            if freq < min_freq:
                break
            if token not in self.stoi:
                idx = len(self.stoi)
                self.stoi[token] = idx
                self.itos[idx] = token

    def __len__(self):
        return len(self.stoi)

    def lookup_token(self, idx: int) -> str:
        return self.itos.get(idx, UNK_TOKEN)

    def lookup_indices(self, tokens: List[str]) -> List[int]:
        return [self.stoi.get(t, UNK_IDX) for t in tokens]


# ──────────────────────────────────────────────────────────────────────
# Multi30k Dataset
# ──────────────────────────────────────────────────────────────────────

class Multi30kDataset(Dataset):
    """
    Multi30k German→English dataset loaded from local JSONL files.

    Args:
        data_dir     : Directory containing train.jsonl, val.jsonl, test.jsonl
        split        : 'train', 'validation', or 'test'
        src_vocab    : Pre-built source (DE) Vocabulary (None for train split)
        tgt_vocab    : Pre-built target (EN) Vocabulary (None for train split)
        min_freq     : Minimum token frequency to include in vocab (train only)
        max_src_len  : Discard training pairs longer than this (0 = no limit)
        max_tgt_len  : Discard training pairs longer than this (0 = no limit)
    """

    def __init__(
        self,
        data_dir: str = 'multi30k',
        split: str = 'train',
        src_vocab: Optional[Vocabulary] = None,
        tgt_vocab: Optional[Vocabulary] = None,
        min_freq: int = 2,
        max_src_len: int = 100,
        max_tgt_len: int = 100,
    ):
        self.split = split

        # Load spaCy tokenizers
        self.spacy_de = spacy.load("de_core_news_sm")
        self.spacy_en = spacy.load("en_core_web_sm")

        # Map split names to actual local filenames
        split_to_file = {
            "train": "train.jsonl",
            "validation": "val.jsonl",
            "test": "test.jsonl"
        }
        
        file_name = split_to_file.get(split, f"{split}.jsonl")
        file_path = os.path.join(data_dir, file_name)

        # Load data from the local JSONL file
        data = []
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Could not find dataset file at {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

        # Tokenize all sentences
        self.src_sentences: List[List[str]] = []
        self.tgt_sentences: List[List[str]] = []

        for item in data:
            de_tokens = self._tokenize_de(item["de"])
            en_tokens = self._tokenize_en(item["en"])

            # For train, optionally filter very long sequences
            if split == "train" and max_src_len > 0 and max_tgt_len > 0:
                if len(de_tokens) > max_src_len or len(en_tokens) > max_tgt_len:
                    continue

            self.src_sentences.append(de_tokens)
            self.tgt_sentences.append(en_tokens)

        # Build vocab from training data or use provided vocab
        if split == "train":
            self.src_vocab = Vocabulary()
            self.tgt_vocab = Vocabulary()
            self.build_vocab(min_freq)
        else:
            assert src_vocab is not None and tgt_vocab is not None, \
                "Must pass pre-built vocabs for validation/test splits"
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab

        # Convert tokens to indices
        self.src_indices, self.tgt_indices = self.process_data()

    def _tokenize_de(self, text: str) -> List[str]:
        return [tok.text.lower() for tok in self.spacy_de.tokenizer(text)]

    def _tokenize_en(self, text: str) -> List[str]:
        return [tok.text.lower() for tok in self.spacy_en.tokenizer(text)]

    def build_vocab(self, min_freq: int = 2):
        """Build src and tgt vocabularies from training data."""
        src_counter = Counter()
        tgt_counter = Counter()
        for tokens in self.src_sentences:
            src_counter.update(tokens)
        for tokens in self.tgt_sentences:
            tgt_counter.update(tokens)
        self.src_vocab.build_from_counter(src_counter, min_freq)
        self.tgt_vocab.build_from_counter(tgt_counter, min_freq)

    def process_data(self) -> Tuple[List[List[int]], List[List[int]]]:
        """Convert token lists to index lists, wrapping with <sos>/<eos>."""
        src_indices = []
        tgt_indices = []
        for src_tokens, tgt_tokens in zip(self.src_sentences, self.tgt_sentences):
            src_idx = [SOS_IDX] + self.src_vocab.lookup_indices(src_tokens) + [EOS_IDX]
            tgt_idx = [SOS_IDX] + self.tgt_vocab.lookup_indices(tgt_tokens) + [EOS_IDX]
            src_indices.append(src_idx)
            tgt_indices.append(tgt_idx)
        return src_indices, tgt_indices

    def __len__(self):
        return len(self.src_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src = torch.tensor(self.src_indices[idx], dtype=torch.long)
        tgt = torch.tensor(self.tgt_indices[idx], dtype=torch.long)
        return src, tgt


# ──────────────────────────────────────────────────────────────────────
# Collate function for DataLoader
# ──────────────────────────────────────────────────────────────────────

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """
    Pad sequences in a batch to the same length.

    Returns:
        src_batch : [batch_size, max_src_len]
        tgt_batch : [batch_size, max_tgt_len]
    """
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_batch, tgt_batch


# ──────────────────────────────────────────────────────────────────────
# Factory: build all three splits + shared vocabs
# ──────────────────────────────────────────────────────────────────────

def build_datasets(data_dir: str = 'multi30k', min_freq: int = 2):
    """
    Build train / val / test datasets with shared vocabularies.

    Returns:
        train_ds, val_ds, test_ds, src_vocab, tgt_vocab
    """
    train_ds = Multi30kDataset(data_dir=data_dir, split="train", min_freq=min_freq)
    val_ds = Multi30kDataset(
        data_dir=data_dir,
        split="validation",
        src_vocab=train_ds.src_vocab,
        tgt_vocab=train_ds.tgt_vocab,
    )
    test_ds = Multi30kDataset(
        data_dir=data_dir,
        split="test",
        src_vocab=train_ds.src_vocab,
        tgt_vocab=train_ds.tgt_vocab,
    )
    return train_ds, val_ds, test_ds, train_ds.src_vocab, train_ds.tgt_vocab


def build_dataloaders(data_dir: str = 'multi30k', batch_size: int = 128, min_freq: int = 2):
    """
    Convenience wrapper: returns DataLoaders and vocab objects.

    Returns:
        train_loader, val_loader, test_loader, src_vocab, tgt_vocab
    """
    train_ds, val_ds, test_ds, src_vocab, tgt_vocab = build_datasets(data_dir, min_freq)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab