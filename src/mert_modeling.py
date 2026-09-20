"""Frozen MERT embedding extraction with split-specific provenance checks.

The tuning notebook calls this module for Train/Validation only. The final
evaluation notebook may call it for Test after all model choices are frozen.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import time

import librosa
import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModel, Wav2Vec2FeatureExtractor

MODEL_NAME = "m-a-p/MERT-v1-95M"
MODEL_REVISION = "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"
SAMPLE_RATE = 24_000
SEGMENT_SECONDS = 10.0
N_LEVELS = 13
HIDDEN_SIZE = 768


def load_frozen_mert(
    device: torch.device | str | None = None,
    model_name: str = MODEL_NAME,
    revision: str = MODEL_REVISION,
):
    """Load the pinned processor and frozen encoder, with MPS to CPU fallback."""
    started = time.perf_counter()
    config = AutoConfig.from_pretrained(
        model_name, revision=revision, trust_remote_code=True
    )
    config.conv_pos_batch_norm = (
        False  # Same remote-code compatibility setting as notebook 17.
    )
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_name, revision=revision, trust_remote_code=True
    )
    encoder = AutoModel.from_pretrained(
        model_name, revision=revision, config=config, trust_remote_code=True
    )
    if (
        int(processor.sampling_rate) != SAMPLE_RATE
        or int(config.hidden_size) != HIDDEN_SIZE
        or int(config.num_hidden_layers) != N_LEVELS - 1
    ):
        raise ValueError(
            "MERT processor/config differs from the pinned input specification"
        )
    encoder.eval()
    # encoder 가중치를 고정하고 뒤쪽 분류기 학습과 표현 추출을 분리한다.
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else (
                "mps"
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                else "cpu"
            )
        )
    device = torch.device(device)
    try:
        encoder.to(device)
    except RuntimeError:
        if device.type != "mps":
            raise
        device = torch.device("cpu")
        encoder.to(device)
    return processor, encoder, device, time.perf_counter() - started


def load_segment_waveform(
    row: pd.Series,
    project_root: str | Path,
    sampling_rate: int = SAMPLE_RATE,
    segment_seconds: float = SEGMENT_SECONDS,
) -> np.ndarray:
    """Read the existing 10-second interval without padding its shorter tail."""
    audio_path = Path(project_root) / str(row["audio_path"])
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    waveform, _ = librosa.load(
        audio_path,
        sr=sampling_rate,
        mono=True,
        offset=float(row["start_sec"]),
        duration=segment_seconds,
    )
    waveform = np.asarray(
        waveform[: int(sampling_rate * segment_seconds)], dtype=np.float32
    )
    if (
        not 0 < len(waveform) <= int(sampling_rate * segment_seconds)
        or not np.isfinite(waveform).all()
    ):
        raise ValueError(f"Invalid waveform for segment {row['segment_id']}")
    return waveform


def extract_layer_embeddings(
    waveform: np.ndarray,
    processor,
    encoder,
    device: torch.device,
    sampling_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Average only valid hidden-state frames for all 13 representation levels."""
    inputs = processor(
        waveform, sampling_rate=sampling_rate, padding=False, return_tensors="pt"
    )
    values = inputs["input_values"].to(device)
    mask = inputs.get("attention_mask")
    if mask is not None:
        # One segment per call and padding=False mean every returned input sample is valid.
        if not bool(mask.bool().all()):
            raise ValueError(
                "Unexpected processor padding in a single-segment MERT call"
            )
        mask = mask.to(device)
    encoder.eval()
    with torch.inference_mode():
        output = encoder(
            input_values=values,
            attention_mask=mask,
            output_hidden_states=True,
            return_dict=True,
        )
    if output.hidden_states is None or len(output.hidden_states) != N_LEVELS:
        raise ValueError("Expected 13 MERT representation levels")
    # 짧은 마지막 Segment를 0으로 채우지 않아 실제 소리의 frame만 평균한다.
    pooled = []
    for hidden in output.hidden_states:
        if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[2] != HIDDEN_SIZE:
            raise ValueError(f"Unexpected MERT hidden shape: {tuple(hidden.shape)}")
        # There is no input padding, so all encoder frames belong to actual audio.
        pooled.append(hidden[0].float().mean(dim=0))
    result = torch.stack(pooled).cpu().numpy()
    if result.shape != (N_LEVELS, HIDDEN_SIZE) or not np.isfinite(result).all():
        raise ValueError("Invalid MERT embedding")
    return result


def embedding_cache_metadata(
    frame: pd.DataFrame,
    manifest_path: str | Path,
    model_name: str = MODEL_NAME,
    revision: str = MODEL_REVISION,
) -> dict:
    """Describe the exact segment IDs, model revision and valid-frame inputs."""
    if (
        frame.empty
        or frame["segment_id"].isna().any()
        or frame["segment_id"].duplicated().any()
    ):
        raise ValueError("Cache frame needs unique, non-null segment IDs")
    return {
        "model": model_name,
        "revision": revision,
        "processor_sampling_rate": SAMPLE_RATE,
        "segment_sec": SEGMENT_SECONDS,
        "input": "24kHz mono, original 10s boundary, no zero padding",
        "pooling": "single-segment valid hidden frames mean",
        "levels": list(range(N_LEVELS)),
        "hidden_size": HIDDEN_SIZE,
        "manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
        "splits": sorted(frame["split"].dropna().unique().tolist()),
        "segment_ids_sha256": hashlib.sha256(
            "\n".join(frame["segment_id"].astype(str)).encode()
        ).hexdigest(),
    }


# 입력: 순서가 고정된 Segment 표와 frozen MERT encoder, cache 경로.
# 출력: 13×768 표현 배열, 사용 장치·시간·파일 경로·출처 메타데이터.
def extract_embedding_cache(
    frame: pd.DataFrame,
    project_root: str | Path,
    manifest_path: str | Path,
    cache_dir: str | Path,
    cache_tag: str,
    processor,
    encoder,
    device: torch.device,
    model_name: str = MODEL_NAME,
    revision: str = MODEL_REVISION,
    checkpoint_every: int = 50,
):
    """Resume a provenance-checked split cache and return embeddings and timing."""
    required = {
        "segment_id",
        "track_sample_id",
        "original_audio",
        "split",
        "audio_path",
        "start_sec",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Cache frame missing columns: {sorted(missing)}")
    if not cache_tag or "/" in cache_tag or "\\" in cache_tag:
        raise ValueError("cache_tag must be a simple filename component")
    frame = frame.reset_index(drop=True)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = f"mert95m_{cache_tag}_validframe_{revision[:12]}"
    paths = {
        kind: cache_dir / f"{stem}{suffix}"
        for kind, suffix in {
            "embeddings": ".npy",
            "done": "_done.npy",
            "index": "_index.csv",
            "metadata": "_meta.json",
        }.items()
    }
    # cache에는 입력 ID 순서와 모델 revision을 함께 기록해 혼용을 막는다.
    metadata = embedding_cache_metadata(frame, manifest_path, model_name, revision)
    expected_shape = (len(frame), N_LEVELS, HIDDEN_SIZE)
    existing = [path.exists() for path in paths.values()]
    if any(existing):
        if not all(existing):
            raise ValueError("Partial MERT cache exists; inspect before reuse")
        if json.loads(paths["metadata"].read_text(encoding="utf-8")) != metadata:
            raise ValueError("MERT cache provenance mismatch")
        index = pd.read_csv(paths["index"])
        if (
            index["segment_id"].astype(str).tolist()
            != frame["segment_id"].astype(str).tolist()
        ):
            raise ValueError("MERT cache segment order mismatch")
        embeddings = np.lib.format.open_memmap(paths["embeddings"], mode="r+")
        done = np.load(paths["done"])
        if embeddings.shape != expected_shape or done.shape != (len(frame),):
            raise ValueError("MERT cache shape mismatch")
    else:
        embeddings = np.lib.format.open_memmap(
            paths["embeddings"], mode="w+", dtype=np.float16, shape=expected_shape
        )
        done = np.zeros(len(frame), dtype=bool)
        frame[["segment_id", "track_sample_id", "original_audio", "split"]].to_csv(
            paths["index"], index=False, encoding="utf-8-sig"
        )
        paths["metadata"].write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        np.save(paths["done"], done)
    started = time.perf_counter()
    # 완료 표시가 없는 Segment만 계산해 중단된 추출을 이어서 수행한다.
    for index in np.flatnonzero(~done):
        row = frame.iloc[int(index)]
        waveform = load_segment_waveform(row, project_root)
        try:
            embedding = extract_layer_embeddings(waveform, processor, encoder, device)
        except RuntimeError as exc:
            if device.type != "mps":
                raise RuntimeError(
                    f"MERT extraction failed for {row['segment_id']}"
                ) from exc
            print("MPS failure; retrying on CPU:", row["segment_id"], str(exc)[:200])
            device = torch.device("cpu")
            encoder.to(device)
            embedding = extract_layer_embeddings(waveform, processor, encoder, device)
        embeddings[index] = embedding.astype(np.float16)
        done[index] = True
        if (int(index) + 1) % checkpoint_every == 0 or int(index) + 1 == len(frame):
            embeddings.flush()
            np.save(paths["done"], done)
            print("MERT cache:", int(done.sum()), "/", len(done))
    embeddings.flush()
    np.save(paths["done"], done)
    if not done.all() or embeddings.shape != expected_shape:
        raise RuntimeError("MERT embedding cache incomplete")
    return embeddings, device, time.perf_counter() - started, paths, metadata
