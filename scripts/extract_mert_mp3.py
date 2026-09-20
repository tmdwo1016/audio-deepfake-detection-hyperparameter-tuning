"""Cache frozen MERT embeddings for both fixed paired MP3 Test conditions.

Run in its own process: this environment can crash when the remote MERT model
is loaded after sklearn. This script does not import sklearn.
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.mert_modeling import (
    MODEL_NAME,
    MODEL_REVISION,
    embedding_cache_metadata,
    extract_layer_embeddings,
    load_frozen_mert,
)
from src.mp3_robustness_inputs import (
    ALL_CONDITIONS,
    fixed_test_manifest,
    load_clean_track,
    load_track,
    slice_segment,
    validate_paired_mp3,
)


def cache_paths(root: Path, condition: str) -> dict[str, Path]:
    directory = root / "data/processed/model_robustness/mp3/mert"
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"mert95m_{condition}_validframe_{MODEL_REVISION[:12]}"
    return {
        "embeddings": directory / f"{stem}.npy",
        "done": directory / f"{stem}_done.npy",
        "index": directory / f"{stem}_index.csv",
        "metadata": directory / f"{stem}_meta.json",
        "summary": directory / f"{stem}_run.json",
    }


# 입력: 조건별 cache 파일 경로·Test 순서·출처 정보.
# 출력: 재개 가능한 임베딩 배열과 완료 표시 배열.
def open_cache(paths: dict[str, Path], test: pd.DataFrame, metadata: dict):
    """Resume only a cache bound to the exact manifest, MP3 files and model."""
    main = [paths[key] for key in ("embeddings", "done", "index", "metadata")]
    existing = [path.exists() for path in main]
    expected_shape = (len(test), 13, 768)
    # 재개할 때는 ID 순서·manifest·압축 보고서가 같은 cache만 허용한다.
    if any(existing):
        if not all(existing):
            raise ValueError("Partial MERT MP3 cache files exist")
        if json.loads(paths["metadata"].read_text(encoding="utf-8")) != metadata:
            raise ValueError("MERT MP3 cache provenance differs")
        index = pd.read_csv(paths["index"])
        if (
            index["segment_id"].astype(str).tolist()
            != test["segment_id"].astype(str).tolist()
        ):
            raise ValueError("MERT MP3 cache ID order differs")
        embeddings = np.lib.format.open_memmap(paths["embeddings"], mode="r+")
        done = np.load(paths["done"])
        if embeddings.shape != expected_shape or done.shape != (len(test),):
            raise ValueError("MERT MP3 cache shape differs")
    else:
        embeddings = np.lib.format.open_memmap(
            paths["embeddings"],
            mode="w+",
            dtype=np.float16,
            shape=expected_shape,
        )
        done = np.zeros(len(test), dtype=bool)
        test[["segment_id", "track_sample_id", "original_audio", "label_id"]].to_csv(
            paths["index"],
            index=False,
            encoding="utf-8-sig",
        )
        paths["metadata"].write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        np.save(paths["done"], done)
    return embeddings, done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=ALL_CONDITIONS,
        default=ALL_CONDITIONS,
        help="Paired conditions to extract in this process",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    test, manifest_path = fixed_test_manifest(root)
    _, transcode_report_hash = validate_paired_mp3(root, test)
    expected = {}
    active = []
    # 조건별 파일을 분리해 clean과 두 bitrate의 표현이 섞이지 않게 한다.
    for condition in args.conditions:
        paths = cache_paths(root, condition)
        metadata = {
            **embedding_cache_metadata(test, manifest_path),
            "condition": condition,
            "transcode_report_sha256": transcode_report_hash,
            "decode": (
                "full track from start, 24kHz mono, original segment start_sec"
                if condition == "clean"
                else "full MP3 from start, 24kHz mono, original segment start_sec"
            ),
            "padding": "none for frozen MERT; valid hidden frames mean",
        }
        embeddings, done = open_cache(paths, test, metadata)
        expected[condition] = (paths, metadata, embeddings, done)
        if not bool(done.all()):
            active.append(condition)
        print(f"MERT {condition}: {int(done.sum())}/{len(done)} cached", flush=True)
    if not active:
        return

    # CPU and one segment per forward pass keep RAM use bounded during CNN work.
    processor, encoder, device, load_seconds = load_frozen_mert(
        device="cpu",
        model_name=MODEL_NAME,
        revision=MODEL_REVISION,
    )
    if str(device) != "cpu":
        raise ValueError("Paired MP3 MERT extraction must run on CPU")
    # 이미 완료된 조건을 건너뛰고 CPU에서 미완료 Segment만 추출한다.
    for condition in active:
        paths, metadata, embeddings, done = expected[condition]
        started = time.perf_counter()
        for track_id, group in test.groupby("track_sample_id", sort=False):
            indices = group.index.to_numpy(dtype=int)
            if bool(done[indices].all()):
                continue
            waveform = (
                load_clean_track(root, group)
                if condition == "clean"
                else load_track(root, condition, str(track_id))
            )
            for index in indices:
                if done[index]:
                    continue
                row = test.iloc[int(index)]
                # MERT는 짧은 끝 구간에 0을 채우지 않고 실제 frame만 평균한다.
                segment = slice_segment(waveform, row["start_sec"], pad=False)
                embedding = extract_layer_embeddings(
                    segment, processor, encoder, device
                )
                embeddings[index] = embedding.astype(np.float16)
                done[index] = True
                if int(done.sum()) % 50 == 0 or bool(done.all()):
                    embeddings.flush()
                    np.save(paths["done"], done)
                    print(
                        f"MERT {condition}: {int(done.sum())}/{len(done)}", flush=True
                    )
        embeddings.flush()
        np.save(paths["done"], done)
        if not bool(done.all()):
            raise RuntimeError(f"MERT {condition} cache incomplete")
        summary = {
            "condition": condition,
            "model_load_seconds": load_seconds,
            "embedding_extraction_seconds": time.perf_counter() - started,
            "device": str(device),
            "shape": list(embeddings.shape),
            "metadata": metadata,
            "paths": {key: str(path) for key, path in paths.items()},
        }
        paths["summary"].write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"MERT {condition} complete: {summary['embedding_extraction_seconds']:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
