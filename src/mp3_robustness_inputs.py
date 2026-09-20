"""Fixed paired MP3 inputs for the optimized binary robustness experiment.

This module deliberately has no sklearn imports so frozen MERT extraction can
run in a separate process from sklearn inference.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

CONDITIONS = {"mp3_128": "128k", "mp3_64": "64k"}
ALL_CONDITIONS = ("clean", *CONDITIONS)
SAMPLE_RATE = 24_000
SEGMENT_SAMPLES = 240_000


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fixed_test_manifest(root: str | Path) -> tuple[pd.DataFrame, Path]:
    """Return existing Test segments in manifest order without changing the split."""
    path = Path(root) / "data/metadata/segment_manifest_10s.csv"
    manifest = pd.read_csv(path)
    # 기존 Test ID만 사용하고 새 split을 만들어 압축 조건에 배정하지 않는다.
    test = manifest.loc[manifest["split"].eq("test")].reset_index(drop=True)
    if len(test) != 1572 or test["track_sample_id"].nunique() != 539:
        raise ValueError("The fixed Test segment/track count changed")
    if test["segment_id"].isna().any() or test["segment_id"].duplicated().any():
        raise ValueError("Test segment IDs are missing or duplicated")
    if (
        test.groupby("track_sample_id")[["original_audio", "label_id", "audio_path"]]
        .nunique()
        .gt(1)
        .any()
        .any()
    ):
        raise ValueError("Test track metadata conflicts")
    if not set(test["label_id"]) == {0, 1}:
        raise ValueError("Both fixed Test classes are required")
    return test, path


def validate_paired_mp3(
    root: str | Path, test: pd.DataFrame
) -> tuple[pd.DataFrame, str]:
    """Require one existing 128k and 64k MP3 for every fixed Test track."""
    root = Path(root)
    report_path = root / "results/mp3_robustness/transcode_report.csv"
    report = pd.read_csv(report_path)
    tracks = set(test["track_sample_id"].astype(str))
    if (
        len(report) != 2 * len(tracks)
        or report.duplicated(["track_sample_id", "condition"]).any()
    ):
        raise ValueError("Paired MP3 transcode report is incomplete or duplicated")
    if set(report["track_sample_id"].astype(str)) != tracks:
        raise ValueError("MP3 report track IDs differ from fixed Test")
    if not report["ok"].eq(True).all():
        raise ValueError("MP3 report contains failed transcodes")
    original = test.drop_duplicates("track_sample_id")[
        [
            "track_sample_id",
            "audio_path",
            "label",
        ]
    ]
    # 압축 파일의 원본 경로·라벨이 고정 manifest와 같은지 대조한다.
    paired = report.merge(
        original,
        on="track_sample_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_manifest"),
    )
    if (
        not paired["source_path"].eq(paired["audio_path"]).all()
        or not paired["label"].eq(paired["label_manifest"]).all()
    ):
        raise ValueError("MP3 source path/label differs from the fixed Test manifest")
    for condition, bitrate in CONDITIONS.items():
        part = report.loc[report["condition"].eq(condition)]
        if len(part) != len(tracks) or set(part["bitrate"]) != {bitrate}:
            raise ValueError(f"{condition} bitrate/track coverage differs")
        for row in part.itertuples(index=False):
            expected = (
                Path("data/processed/mp3_robustness")
                / condition
                / f"{row.track_sample_id}.mp3"
            )
            if Path(row.compressed_path) != expected:
                raise ValueError(f"Unexpected MP3 path for {row.track_sample_id}")
            path = root / expected
            if not path.is_file() or path.stat().st_size < 1024:
                raise FileNotFoundError(path)
    return report, sha256_file(report_path)


def audio_path(root: str | Path, condition: str, track_id: str) -> Path:
    if condition not in CONDITIONS:
        raise ValueError(f"Unsupported predeclared condition: {condition}")
    return Path(root) / "data/processed/mp3_robustness" / condition / f"{track_id}.mp3"


def load_track(root: str | Path, condition: str, track_id: str) -> np.ndarray:
    """Decode one complete MP3 from the beginning, avoiding inaccurate seek offsets."""
    path = audio_path(root, condition, track_id)
    # 곡 처음부터 전체 디코딩해야 clean과 MP3의 시작 위치가 같은 기준을 쓴다.
    waveform, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim != 1 or not len(waveform) or not np.isfinite(waveform).all():
        raise ValueError(f"Invalid decoded audio: {path}")
    return waveform


def load_clean_track(root: str | Path, group: pd.DataFrame) -> np.ndarray:
    """Decode the original Test track fully with the same path used for MP3."""
    if group["audio_path"].nunique() != 1:
        raise ValueError("A clean Test track has more than one audio path")
    path = Path(root) / str(group.iloc[0]["audio_path"])
    # 곡 처음부터 전체 디코딩해야 clean과 MP3의 시작 위치가 같은 기준을 쓴다.
    waveform, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim != 1 or not len(waveform) or not np.isfinite(waveform).all():
        raise ValueError(f"Invalid decoded clean audio: {path}")
    return waveform


def slice_segment(waveform: np.ndarray, start_sec: float, *, pad: bool) -> np.ndarray:
    """Use the original 10-second boundary, padding only models that require it."""
    # 세 조건 모두 원래 시작 시각을 표본 위치로 바꿔 동일 구간을 자른다.
    start = int(round(float(start_sec) * SAMPLE_RATE))
    segment = np.asarray(waveform[start : start + SEGMENT_SAMPLES], dtype=np.float32)
    if not 0 < len(segment) <= SEGMENT_SAMPLES or not np.isfinite(segment).all():
        raise ValueError(f"Invalid segment at {start_sec} seconds")
    if pad and len(segment) < SEGMENT_SAMPLES:
        segment = np.pad(segment, (0, SEGMENT_SAMPLES - len(segment)))
    return segment
