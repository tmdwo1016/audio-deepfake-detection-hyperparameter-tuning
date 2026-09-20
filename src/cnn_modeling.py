"""Reusable Log-Mel CNN architecture and inference for the revised binary protocol."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset


class LogMelDataset(Dataset):
    """Read the existing Log-Mel cache without changing segment boundaries."""

    def __init__(self, metadata: pd.DataFrame, cache_path: Path, index: pd.DataFrame):
        self.metadata = metadata.reset_index(drop=True).copy()
        self.cache_path = Path(cache_path)
        # 배열 위치 대신 segment_id로 찾으면 cache 순서와 평가 순서가 달라도 안전하다.
        self.index_by_id = dict(
            zip(index["segment_id"], index["logmel_index"], strict=True)
        )
        if len(self.index_by_id) != len(index):
            raise ValueError("Log-Mel index has duplicate segment_id")
        missing = set(self.metadata["segment_id"]) - set(self.index_by_id)
        if missing:
            raise ValueError(f"Log-Mel cache lacks {len(missing)} segments")
        self.cache = np.load(self.cache_path, mmap_mode="r")
        if len(self.cache) != len(index):
            raise ValueError("Log-Mel cache and index row counts differ")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        """Return one [1, Mel bins, time frames] tensor, binary label, and segment ID."""
        row = self.metadata.iloc[index]
        cache_index = int(self.index_by_id[row["segment_id"]])
        # float16 cache is converted to float32 for stable convolution arithmetic.
        array = np.array(self.cache[cache_index], dtype=np.float32, copy=True)
        if array.ndim != 2 or not np.isfinite(array).all():
            raise ValueError(f"Invalid Log-Mel tensor for {row['segment_id']}")
        tensor = torch.from_numpy(array).unsqueeze(0)
        return tensor, torch.tensor(float(row["label_id"])), row["segment_id"]


class ConvBlock(nn.Module):
    """Preserve the two-convolution plus pooling block of the historical CNN."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # 두 합성곱으로 특징을 뽑고 pooling으로 시간·주파수 크기를 절반으로 줄인다.
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, tensor):
        """Apply convolutions, normalization, activation, and spatial pooling."""
        return self.block(tensor)


class LogMelCNN(nn.Module):
    """Use the historical 1→16→32→64→128 CNN with tunable dropout."""

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, 1)
        )

    def forward(self, tensor):
        """Convert each Log-Mel image into one FAKE logit."""
        # [B, 1, 128, T] → [B, 16, 64, floor(T/2)]
        # → [B, 32, 32, floor(T/4)] → [B, 64, 16, floor(T/8)]
        # → [B, 128, 8, floor(T/16)] with four ConvBlocks.
        tensor = self.features(tensor)
        # [B, 128, 8, ...] → [B, 128, 1, 1].
        tensor = self.pool(tensor)
        # Flatten [B, 128] → Linear [B, 1] → [B].
        return self.classifier(tensor).squeeze(1)


@torch.no_grad()
def predict_scores(model, loader, metadata: pd.DataFrame, device):
    """Return standardized segment scores using sigmoid(logit), never calibrated probabilities."""
    model.eval()
    # batch 추론 순서에 기대지 않고 ID별 FAKE 점수를 다시 연결한다.
    score_by_id = {}
    for tensors, labels, segment_ids in loader:
        logits = model(tensors.to(device))
        scores = torch.sigmoid(logits).cpu().numpy()
        for segment_id, label, score in zip(
            segment_ids, labels.numpy(), scores, strict=True
        ):
            if segment_id in score_by_id:
                raise ValueError(f"Duplicate predicted segment_id: {segment_id}")
            score_by_id[segment_id] = (int(label), float(score))
    if set(score_by_id) != set(metadata["segment_id"]):
        raise ValueError("Inference did not cover exactly the requested segment IDs")
    frame = metadata[
        ["segment_id", "track_sample_id", "original_audio", "label_id"]
    ].copy()
    frame = frame.rename(
        columns={
            "track_sample_id": "track_id",
            "original_audio": "original_audio_id",
            "label_id": "label",
        }
    )
    frame["score"] = frame["segment_id"].map(lambda key: score_by_id[key][1])
    if not np.array_equal(
        frame["label"].to_numpy(),
        np.array([score_by_id[key][0] for key in frame["segment_id"]]),
    ):
        raise ValueError("Inference labels differ from the manifest")
    return frame.reset_index(drop=True)
