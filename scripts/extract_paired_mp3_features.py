"""Extract paired clean/MP3 Test features from full-track decoded audio.

Clean and both predeclared MP3 conditions use identical manifest segment start
times. Handcrafted and Log-Mel inputs retain their original 10-second padding
and feature settings. No model or threshold is fit here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", tempfile.gettempdir())
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import librosa
import numpy as np
import pandas as pd

from src.mp3_robustness_inputs import (
    ALL_CONDITIONS,
    fixed_test_manifest,
    load_clean_track,
    load_track,
    sha256_file,
    slice_segment,
    validate_paired_mp3,
)

SR = 24_000
N_FFT = 1024
HOP = 240
N_MELS = 128
N_MFCC = 40
FMAX = 12_000
PROTOCOL_SHA256 = "95f1546ce8e3fcedb66e5ef52488c9255e414965514b5163e1a62b9052e4941e"


def _add_mean_std(features: dict, prefix: str, values: np.ndarray) -> None:
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[np.newaxis, :]
    # 각 음향 계열의 시간별 평균·표준편차를 개별 feature 열로 펼친다.
    for index in range(values.shape[0]):
        name = f"{prefix}_{index + 1:02d}" if values.shape[0] > 1 else prefix
        features[f"{name}_mean"] = float(np.mean(values[index]))
        features[f"{name}_std"] = float(np.std(values[index]))


def handcrafted_features(waveform: np.ndarray) -> dict[str, float]:
    """Reproduce the original 266-D librosa feature extractor exactly."""
    features = {}
    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=SR,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        fmax=FMAX,
    )
    _add_mean_std(features, "mfcc", mfcc)
    _add_mean_std(features, "mfcc_delta", librosa.feature.delta(mfcc, order=1))
    _add_mean_std(features, "mfcc_delta2", librosa.feature.delta(mfcc, order=2))
    for name, values in (
        (
            "spectral_centroid",
            librosa.feature.spectral_centroid(
                y=waveform, sr=SR, n_fft=N_FFT, hop_length=HOP
            ),
        ),
        (
            "spectral_bandwidth",
            librosa.feature.spectral_bandwidth(
                y=waveform, sr=SR, n_fft=N_FFT, hop_length=HOP
            ),
        ),
        (
            "spectral_rolloff",
            librosa.feature.spectral_rolloff(
                y=waveform, sr=SR, n_fft=N_FFT, hop_length=HOP, roll_percent=0.85
            ),
        ),
        (
            "spectral_flatness",
            librosa.feature.spectral_flatness(y=waveform, n_fft=N_FFT, hop_length=HOP),
        ),
        (
            "spectral_contrast",
            librosa.feature.spectral_contrast(
                y=waveform, sr=SR, n_fft=N_FFT, hop_length=HOP
            ),
        ),
        ("rms", librosa.feature.rms(y=waveform, frame_length=N_FFT, hop_length=HOP)),
        (
            "zcr",
            librosa.feature.zero_crossing_rate(
                y=waveform, frame_length=N_FFT, hop_length=HOP
            ),
        ),
    ):
        _add_mean_std(features, name, values)
    if len(features) != 266 or not np.isfinite(list(features.values())).all():
        raise ValueError("Handcrafted extractor did not produce 266 finite values")
    return features


def logmel_features(waveform: np.ndarray) -> np.ndarray:
    """Reproduce the 128×1001 fixed Log-Mel transform and float16 cache."""
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        fmax=FMAX,
        power=2.0,
        center=True,
    )
    # dB 범위를 고정해 기존 CNN이 학습한 Log-Mel 입력 척도를 재현한다.
    decibels = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    result = np.asarray((decibels + 80.0) / 40.0 - 1.0, dtype=np.float16)
    if result.shape != (128, 1001) or not np.isfinite(result).all():
        raise ValueError("Invalid Log-Mel shape or values")
    return result


def open_array(
    path: Path,
    done_path: Path,
    shape: tuple[int, ...],
    dtype,
    metadata_path: Path,
    metadata: dict,
):
    present = [p.exists() for p in (path, done_path, metadata_path)]
    # 부분 cache나 다른 프로토콜의 cache는 이어 쓰지 않는다.
    if any(present):
        if not all(present):
            raise ValueError(f"Partial cache at {path}")
        if json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
            raise ValueError(f"Cache provenance changed at {path}")
        array = np.lib.format.open_memmap(path, mode="r+")
        done = np.load(done_path)
        if (
            array.shape != shape
            or array.dtype != np.dtype(dtype)
            or done.shape != (shape[0],)
        ):
            raise ValueError(f"Cache shape/dtype changed at {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
        done = np.zeros(shape[0], dtype=bool)
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        np.save(done_path, done)
    return array, done


# 입력: clean/128/64 조건, 고정 Test 행, manifest·압축 보고서 해시.
# 출력: 반환값 대신 266차원 특징과 clean Log-Mel의 재개 가능 cache 파일.
def extract_condition(
    root: Path,
    condition: str,
    test: pd.DataFrame,
    manifest_hash: str,
    report_hash: str,
    feature_columns: list[str],
) -> None:
    cache_dir = root / "data/processed/model_robustness/mp3"
    ids_hash = (
        __import__("hashlib")
        .sha256("\n".join(test["segment_id"].astype(str)).encode())
        .hexdigest()
    )
    common = {
        "condition": condition,
        "manifest_sha256": manifest_hash,
        "transcode_report_sha256": report_hash,
        "protocol_sha256": PROTOCOL_SHA256,
        "segment_ids_sha256": ids_hash,
        "decoder": "full track from beginning, 24kHz mono",
        "segment": "original start_sec, 240000 samples, zero-pad short tail",
    }
    feature_path = cache_dir / "handcrafted" / f"{condition}_features.npy"
    feature_done_path = cache_dir / "handcrafted" / f"{condition}_done.npy"
    feature_meta_path = cache_dir / "handcrafted" / f"{condition}_meta.json"
    feature_array, feature_done = open_array(
        feature_path,
        feature_done_path,
        (len(test), 266),
        np.float64,
        feature_meta_path,
        {
            **common,
            "feature_columns": feature_columns,
            "source": "08_extract_handcrafted_features.ipynb",
        },
    )
    logmel_array = logmel_done = None
    if condition == "clean":
        logmel_path = cache_dir / "logmel" / "clean_float16.npy"
        logmel_done_path = cache_dir / "logmel" / "clean_done.npy"
        logmel_meta_path = cache_dir / "logmel" / "clean_meta.json"
        logmel_array, logmel_done = open_array(
            logmel_path,
            logmel_done_path,
            (len(test), 128, 1001),
            np.float16,
            logmel_meta_path,
            {
                **common,
                "settings": {
                    "n_fft": N_FFT,
                    "hop_length": HOP,
                    "n_mels": N_MELS,
                    "fmax": FMAX,
                    "power": 2.0,
                    "center": True,
                    "top_db": 80,
                    "ref": "max",
                    "fixed_scale": "(dB+40)/40",
                },
            },
        )
    print(f"{condition}: handcrafted {int(feature_done.sum())}/{len(test)}", flush=True)
    started = time.perf_counter()
    # 곡 전체를 한 번 디코딩하고 원래 Segment 시작점에서 세 조건을 동일하게 자른다.
    for track_id, group in test.groupby("track_sample_id", sort=False):
        indices = group.index.to_numpy(dtype=int)
        if bool(feature_done[indices].all()) and (
            logmel_done is None or bool(logmel_done[indices].all())
        ):
            continue
        waveform = (
            load_clean_track(root, group)
            if condition == "clean"
            else load_track(root, condition, str(track_id))
        )
        for index in indices:
            if feature_done[index] and (logmel_done is None or logmel_done[index]):
                continue
            row = test.iloc[int(index)]
            segment = slice_segment(waveform, row["start_sec"], pad=True)
            if not feature_done[index]:
                values = handcrafted_features(segment)
                if set(values) != set(feature_columns):
                    raise ValueError(
                        "Handcrafted feature columns differ from trained pipeline"
                    )
                feature_array[index] = [values[name] for name in feature_columns]
                feature_done[index] = True
            if logmel_done is not None and not logmel_done[index]:
                logmel_array[index] = logmel_features(segment)
                logmel_done[index] = True
            if int(feature_done.sum()) % 50 == 0 or bool(feature_done.all()):
                feature_array.flush()
                np.save(feature_done_path, feature_done)
                if logmel_done is not None:
                    logmel_array.flush()
                    np.save(logmel_done_path, logmel_done)
                print(f"{condition}: {int(feature_done.sum())}/{len(test)}", flush=True)
    feature_array.flush()
    np.save(feature_done_path, feature_done)
    if logmel_done is not None:
        logmel_array.flush()
        np.save(logmel_done_path, logmel_done)
    if not bool(feature_done.all()) or (
        logmel_done is not None and not bool(logmel_done.all())
    ):
        raise RuntimeError(f"{condition} paired cache incomplete")
    print(
        f"{condition} feature extraction complete: {time.perf_counter() - started:.1f}s",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--condition", choices=ALL_CONDITIONS, action="append")
    args = parser.parse_args()
    root = args.root.resolve()
    test, manifest_path = fixed_test_manifest(root)
    _, report_hash = validate_paired_mp3(root, test)
    protocol_hash = sha256_file(root / "docs/ROBUSTNESS_PROTOCOL.md")
    if protocol_hash != PROTOCOL_SHA256:
        raise ValueError("Predeclared robustness protocol changed")
    manifest_hash = sha256_file(manifest_path)
    classical_meta = json.loads(
        (root / "results/model_tuning/classical_run_metadata.json").read_text()
    )
    if classical_meta["split_manifest_sha256"] != manifest_hash:
        raise ValueError("Classical checkpoint manifest differs")
    columns = classical_meta["feature_columns"]
    if len(columns) != 266 or len(set(columns)) != 266:
        raise ValueError("Expected 266 ordered handcrafted features")
    # clean도 새로 추출해 MP3와 같은 디코딩 경로의 paired 기준으로 사용한다.
    for condition in args.condition or ALL_CONDITIONS:
        extract_condition(root, condition, test, manifest_hash, report_hash, columns)


if __name__ == "__main__":
    main()
