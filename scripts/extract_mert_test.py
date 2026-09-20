"""Extract frozen MERT Test embeddings in a process without sklearn imports.

In this environment the custom MERT model can segfault when sklearn was
imported earlier in the same process. The final notebook calls this script
only after all Train/Validation selections are frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", tempfile.gettempdir())

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mert_modeling import (
    MODEL_NAME,
    MODEL_REVISION,
    extract_embedding_cache,
    load_frozen_mert,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = root / "data/metadata/segment_manifest_10s.csv"
    # Test를 읽기 전에 동결 당시 manifest와 현재 파일이 같은지 확인한다.
    actual_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if actual_hash != args.manifest_sha256:
        raise ValueError("Frozen manifest hash changed before MERT Test extraction")

    # sklearn/shared evaluator는 이 프로세스에서 불러오지 않는다.
    processor, encoder, device, load_seconds = load_frozen_mert(
        device="cpu", model_name=MODEL_NAME, revision=MODEL_REVISION
    )
    manifest = pd.read_csv(manifest_path)
    # 원래 group split의 Test 행만 골라 layer 표현을 만든다.
    test = manifest.loc[manifest["split"].eq("test")].reset_index(drop=True)
    if test.empty or test["label_id"].nunique() != 2:
        raise ValueError("Fixed Test split is missing or single-class")
    embeddings, device, extraction_seconds, paths, metadata = extract_embedding_cache(
        test,
        root,
        manifest_path,
        root / "data/processed/mert",
        "binary_test",
        processor,
        encoder,
        device,
        model_name=MODEL_NAME,
        revision=MODEL_REVISION,
    )
    summary = {
        "device": str(device),
        "model_load_seconds": load_seconds,
        "embedding_extraction_seconds": extraction_seconds,
        "shape": list(embeddings.shape),
        "paths": {key: str(path) for key, path in paths.items()},
        "metadata": metadata,
    }
    summary_path = paths["metadata"].with_name(paths["metadata"].stem + "_run.json")
    # 추출 장치·시간·cache 경로를 남겨 후속 점수의 출처를 추적한다.
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("MERT Test cache complete:", summary_path)


if __name__ == "__main__":
    main()
