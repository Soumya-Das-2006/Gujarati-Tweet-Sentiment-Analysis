"""
Shared vocabulary, dataset, and model definition for the two-stage ANN
(Embedding + BiLSTM) sentiment classifier used by src/train_ann.py.
"""
from collections import Counter

import torch
from torch import nn
from torch.utils.data import Dataset

PAD_IDX = 0
UNK_IDX = 1


def build_vocab(text_iterable, max_vocab_size: int = 40000) -> dict:
    counter = Counter()
    for text in text_iterable:
        counter.update(str(text).split())
    most_common = counter.most_common(max_vocab_size - 2)
    vocab = {"<pad>": PAD_IDX, "<unk>": UNK_IDX}
    for i, (word, _) in enumerate(most_common, start=2):
        vocab[word] = i
    return vocab


def encode(text: str, vocab: dict, max_len: int) -> list[int]:
    ids = [vocab.get(tok, UNK_IDX) for tok in str(text).split()[:max_len]]
    ids += [PAD_IDX] * (max_len - len(ids))
    return ids


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab: dict, label2id: dict, max_len: int = 40):
        self.texts = list(texts)
        self.labels = [label2id[l] for l in labels]
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = encode(self.texts[idx], self.vocab, self.max_len)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


class BiLSTMClassifier(nn.Module):
    """Embedding -> BiLSTM -> masked mean pooling -> Dense -> swappable head."""

    def __init__(self, vocab_size: int, embed_dim: int = 100, hidden_dim: int = 128, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(hidden_dim * 2, 64)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(64, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        mask = (x != PAD_IDX).unsqueeze(-1).float()  # (B, T, 1)
        embedded = self.embedding(x)  # (B, T, E)
        lstm_out, _ = self.lstm(embedded)  # (B, T, 2H)
        summed = (lstm_out * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / counts  # masked mean pooling
        return self.dropout(torch.relu(self.dense(pooled)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x))

    def replace_head(self, num_classes: int):
        in_features = self.classifier.in_features
        self.classifier = nn.Linear(in_features, num_classes)
        return self
