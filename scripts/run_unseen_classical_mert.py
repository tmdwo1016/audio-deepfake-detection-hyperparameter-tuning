"""Strict generator holdout search for handcrafted LR/SVM and frozen MERT+LR.

MusicGen and Udio are each removed from both Train and Validation before any
candidate selection. Original Test groups are opened only after both searches
have selected and saved their models and Validation thresholds.
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.modeling_evaluation import (
    assert_group_disjoint,
    evaluate_binary_predictions,
    select_best_candidate,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data/metadata/segment_manifest_10s.csv"
FEATURE_PATH = ROOT / "data/processed/features/handcrafted_features_10s.csv"
MERT_DIR = ROOT / "data/processed/mert"
MERT_REVISION = "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"
MERT_TRAIN_STEM = f"mert95m_binary_train_val_validframe_{MERT_REVISION[:12]}"
MERT_TEST_STEM = f"mert95m_binary_test_validframe_{MERT_REVISION[:12]}"
RESULT_ROOT = ROOT / "results/unseen_revised_classical_mert"
CHECKPOINT_ROOT = ROOT / "checkpoints/unseen_revised_classical_mert"
SEED = 42
TARGETS = ("musicgen", "udio")
LR_C = (0.01, 0.1, 1.0, 10.0, 100.0)
SVM_C = (0.1, 1.0, 10.0, 100.0)
SVM_GAMMA = ("scale", 0.001, 0.01, 0.1)
MERT_LAYERS = tuple(range(13))
METADATA_COLUMNS = (
    "segment_id",
    "track_sample_id",
    "original_audio",
    "label",
    "label_id",
    "split",
    "generator",
)


def file_sha256(path: Path) -> str:
    """Hash exact bytes of a source or checkpoint file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def log(message: str) -> None:
    """Print a time-stamped progress line during the full search."""
    print(datetime.now(timezone.utc).isoformat(timespec="seconds"), message, flush=True)


def score_frame(meta: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """Attach FAKE-direction scores to the original segment, track and group IDs."""
    if len(meta) != len(scores):
        raise ValueError("The candidate did not score every requested segment")
    frame = (
        meta[["segment_id", "track_sample_id", "original_audio", "label_id"]]
        .rename(
            columns={
                "track_sample_id": "track_id",
                "original_audio": "original_audio_id",
                "label_id": "label",
            }
        )
        .copy()
    )
    frame["score"] = np.asarray(scores, dtype=float)
    if not np.isfinite(frame["score"]).all():
        raise ValueError("The candidate returned missing or infinite scores")
    return frame


def read_manifest() -> tuple[pd.DataFrame, str]:
    """Validate the original group split and REAL=0/FAKE=1 mapping."""
    # 원래 원곡 단위 split과 REAL=0/FAKE=1 매핑을 그대로 사용한다.
    manifest = pd.read_csv(MANIFEST_PATH)
    if manifest["segment_id"].isna().any() or manifest["segment_id"].duplicated().any():
        raise ValueError("Original manifest has duplicate or null segment IDs")
    if not set(METADATA_COLUMNS).issubset(manifest.columns):
        raise ValueError("Original manifest lacks a required holdout column")
    if not (
        (manifest["label"].eq("REAL") & manifest["label_id"].eq(0))
        | (manifest["label"].eq("FAKE") & manifest["label_id"].eq(1))
    ).all():
        raise ValueError("Original REAL=0/FAKE=1 mapping differs")
    assert_group_disjoint(
        manifest.rename(columns={"original_audio": "original_audio_id"})
    )
    if (
        manifest.groupby("track_sample_id")[["label_id", "original_audio", "split"]]
        .nunique()
        .gt(1)
        .any()
        .any()
    ):
        raise ValueError("Track label, group, or split is inconsistent")
    if set(manifest["split"].unique()) != {"train", "val", "test"}:
        raise ValueError("Original split set differs")
    return manifest, file_sha256(MANIFEST_PATH)


# 후보 선택 중에는 Train·Validation 특징만 읽고 Test 특징 행은 제외한다.
def read_training_features(
    manifest: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    """Load only Train/Validation feature rows before the Test evaluation stage."""
    chunks = []
    for chunk in pd.read_csv(FEATURE_PATH, chunksize=3000):
        chunks.append(chunk.loc[chunk["split"].isin(("train", "val"))].copy())
    features = pd.concat(chunks, ignore_index=True)
    expected = manifest.loc[manifest["split"].isin(("train", "val"))].copy()
    if len(feature_columns) != 266 or not set(feature_columns).issubset(
        features.columns
    ):
        raise ValueError("Saved handcrafted feature schema differs from 266-D")
    left = (
        features[list(METADATA_COLUMNS)]
        .sort_values("segment_id")
        .reset_index(drop=True)
    )
    right = (
        expected[list(METADATA_COLUMNS)]
        .sort_values("segment_id")
        .reset_index(drop=True)
    )
    if not left.equals(right):
        raise ValueError(
            "Train/Validation feature metadata differs from segment manifest"
        )
    features[feature_columns] = features[feature_columns].apply(
        pd.to_numeric, errors="raise"
    )
    if not np.isfinite(features[feature_columns].to_numpy(dtype=np.float64)).all():
        raise ValueError("Train/Validation feature matrix contains invalid values")
    return features


def read_mert_cache(stem: str, expected: pd.DataFrame, manifest_hash: str):
    """Verify frozen valid-frame cache provenance and return its memory map."""
    index_path = MERT_DIR / f"{stem}_index.csv"
    metadata_path = MERT_DIR / f"{stem}_meta.json"
    done_path = MERT_DIR / f"{stem}_done.npy"
    embeddings_path = MERT_DIR / f"{stem}.npy"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata["model"] != "m-a-p/MERT-v1-95M"
        or metadata["revision"] != MERT_REVISION
        or metadata["processor_sampling_rate"] != 24000
        or metadata["pooling"] != "single-segment valid hidden frames mean"
        or metadata["manifest_sha256"] != manifest_hash
    ):
        raise ValueError(f"MERT cache provenance differs: {stem}")
    expected_ids = expected["segment_id"].astype(str).tolist()
    cached_ids = pd.read_csv(index_path)["segment_id"].astype(str).tolist()
    if (
        cached_ids != expected_ids
        or metadata["segment_ids_sha256"]
        != hashlib.sha256("\n".join(expected_ids).encode()).hexdigest()
    ):
        raise ValueError(f"MERT cache segment order differs: {stem}")
    if not bool(np.load(done_path).all()):
        raise ValueError(f"MERT cache has unfinished segments: {stem}")
    embeddings = np.load(embeddings_path, mmap_mode="r")
    if embeddings.shape != (len(expected), 13, 768):
        raise ValueError(f"Unexpected MERT embedding shape: {embeddings.shape}")
    return embeddings, metadata


# 목표 생성기의 FAKE만 기존 Train 또는 Validation에서 제거한다.
def filtered_split(meta: pd.DataFrame, target: str, split: str) -> pd.DataFrame:
    """Keep every REAL and every other-generator FAKE in the original split."""
    selected = (
        meta.loc[
            meta["split"].eq(split)
            & (meta["label_id"].eq(0) | meta["generator"].ne(target))
        ]
        .copy()
        .reset_index(drop=True)
    )
    if selected["label_id"].nunique() != 2:
        raise ValueError(f"{target} {split} lacks a class after exclusion")
    if selected.loc[selected["label_id"].eq(1), "generator"].eq(target).any():
        raise ValueError(f"{target} leaked into {split} candidates")
    return selected


def score_lr(model: Pipeline, matrix: pd.DataFrame) -> np.ndarray:
    """Choose the FAKE=1 probability column from a fitted LR Pipeline."""
    classifier = model.named_steps["classifier"]
    if not np.array_equal(classifier.classes_, [0, 1]):
        raise ValueError("LR classes are not REAL=0, FAKE=1")
    return model.predict_proba(matrix)[
        :, int(np.flatnonzero(classifier.classes_ == 1)[0])
    ]


def select_row(rows: list[dict]) -> dict:
    """Use the shared unrounded Track EER, AUC, candidate-order rule."""
    return select_best_candidate(rows)


def save_validation_scores(
    report: dict,
    frame: pd.DataFrame,
    target: str,
    model_name: str,
    run_id: str,
    output: Path,
) -> None:
    """Save selected Validation segment and track scores with separate thresholds."""
    for level in ("segment", "track"):
        raw = report[f"{level}_scores"].copy()
        raw.insert(0, "threshold", report["thresholds"][level])
        raw.insert(0, "level", level)
        raw.insert(0, "split", "val")
        raw.insert(0, "model", model_name)
        raw.insert(0, "holdout_generator", target)
        raw.insert(0, "run_id", run_id)
        raw.to_csv(
            output
            / f"{target}_{model_name.lower().replace(' ', '_')}_{level}_val_scores.csv",
            index=False,
            encoding="utf-8-sig",
        )


# 입력: 대상 생성기를 뺀 Train/Validation과 266차원 특징.
# 출력: 후보별 Validation 기록과 선택 checkpoint·임계값 정보.
def train_classical(
    target: str,
    model_name: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: pd.DataFrame,
    feature_columns: list[str],
    checkpoint: Path,
    run_id: str,
    output: Path,
) -> tuple[list[dict], dict]:
    """Search every classical candidate with the target absent from Train/Val."""
    x_train = train[["segment_id"]].merge(
        features, on="segment_id", validate="one_to_one"
    )[feature_columns]
    x_val = val[["segment_id"]].merge(features, on="segment_id", validate="one_to_one")[
        feature_columns
    ]
    if len(x_train) != len(train) or len(x_val) != len(val):
        raise ValueError("Classical candidate has missing feature segments")
    y_train = train["label_id"].to_numpy(dtype=int)
    candidates = (
        [{"C": c} for c in LR_C]
        if model_name == "LogisticRegression"
        else [{"C": c, "gamma": gamma} for c in SVM_C for gamma in SVM_GAMMA]
    )
    rows: list[dict] = []
    selected: dict | None = None
    for order, config in enumerate(candidates):
        classifier = (
            LogisticRegression(
                C=config["C"],
                solver="lbfgs",
                penalty="l2",
                class_weight="balanced",
                max_iter=3000,
                random_state=SEED,
            )
            if model_name == "LogisticRegression"
            else SVC(
                kernel="rbf",
                C=config["C"],
                gamma=config["gamma"],
                class_weight="balanced",
                probability=False,
                cache_size=2048,
                random_state=SEED,
            )
        )
        # 표준화와 분류기를 함께 학습해 Validation/Test에 새 scaler를 맞추지 않는다.
        model = Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x_train, y_train)
        fit_sec = time.perf_counter() - started
        started = time.perf_counter()
        if model_name == "LogisticRegression":
            scores = score_lr(model, x_val)
        else:
            if not np.array_equal(classifier.classes_, [0, 1]):
                raise ValueError("SVM classes are not REAL=0, FAKE=1")
            scores = model.decision_function(x_val)
        inference_sec = time.perf_counter() - started
        frame = score_frame(val, scores)
        report = evaluate_binary_predictions(frame)
        row = {
            "run_id": run_id,
            "holdout_generator": target,
            "model": model_name,
            "candidate_order": order,
            "C": config["C"],
            "gamma": config.get("gamma"),
            "layer": None,
            "train_segments": len(train),
            "val_segments": len(val),
            "val_segment_auc": report["segment"]["roc_auc"],
            "val_segment_eer": report["segment"]["eer"],
            "track_roc_auc": report["track"]["roc_auc"],
            "track_eer": report["track"]["eer"],
            "val_track_macro_f1": report["track"]["macro_f1"],
            "fit_seconds": fit_sec,
            "val_inference_seconds": inference_sec,
            "warnings": json.dumps(
                [str(item.message) for item in captured], ensure_ascii=False
            ),
        }
        rows.append(row)
        if select_row(rows)["candidate_order"] == order:
            joblib.dump(model, checkpoint)
            selected = {
                "row": row.copy(),
                "thresholds": report["thresholds"],
                "checkpoint": str(checkpoint),
                "validation_frame": frame,
                "validation_report": report,
            }
        log(
            f"{target} {model_name} candidate {order+1}/{len(candidates)} "
            f"Val Track EER={row['track_eer']:.6f} AUC={row['track_roc_auc']:.6f}"
        )
    if (
        selected is None
        or select_row(rows)["candidate_order"] != selected["row"]["candidate_order"]
    ):
        raise RuntimeError(
            "Classical selection does not match the full Validation table"
        )
    save_validation_scores(
        selected["validation_report"],
        selected["validation_frame"],
        target,
        model_name,
        run_id,
        output,
    )
    selected["checkpoint_sha256"] = file_sha256(checkpoint)
    del selected["validation_frame"], selected["validation_report"]
    return rows, selected


# 입력: 같은 필터된 split과 frozen 13-layer MERT 표현.
# 출력: layer·C 후보 기록과 선택된 LR head/임계값 정보.
def train_mert(
    target: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    all_meta: pd.DataFrame,
    embeddings: np.ndarray,
    checkpoint: Path,
    run_id: str,
    output: Path,
) -> tuple[list[dict], dict]:
    """Search all 13 frozen levels and five C values on filtered groups only."""
    positions = pd.Index(all_meta["segment_id"])
    train_positions = positions.get_indexer(train["segment_id"])
    val_positions = positions.get_indexer(val["segment_id"])
    if (train_positions < 0).any() or (val_positions < 0).any():
        raise ValueError("MERT Train/Validation cache lacks a selected segment")
    y_train = train["label_id"].to_numpy(dtype=int)
    rows: list[dict] = []
    selected: dict | None = None
    for layer in MERT_LAYERS:
        # Frozen embeddings are independent of the generator inclusion mask;
        # the scaler is fit only on this holdout's filtered Train segments.
        x_train = np.asarray(embeddings[train_positions, layer, :], dtype=np.float32)
        x_val = np.asarray(embeddings[val_positions, layer, :], dtype=np.float32)
        if not np.isfinite(x_train).all() or not np.isfinite(x_val).all():
            raise ValueError("MERT cache contains missing embeddings")
        # MERT 768차원 표준화도 해당 holdout의 필터된 Train에서만 맞춘다.
        scaler = StandardScaler().fit(x_train)
        z_train = scaler.transform(x_train)
        z_val = scaler.transform(x_val)
        for c in LR_C:
            order = len(rows)
            classifier = LogisticRegression(
                C=c,
                solver="lbfgs",
                penalty="l2",
                class_weight="balanced",
                max_iter=3000,
                random_state=SEED,
            )
            started = time.perf_counter()
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always", ConvergenceWarning)
                classifier.fit(z_train, y_train)
            fit_sec = time.perf_counter() - started
            if not np.array_equal(classifier.classes_, [0, 1]):
                raise ValueError("MERT LR classes are not REAL=0, FAKE=1")
            started = time.perf_counter()
            fake_column = int(np.flatnonzero(classifier.classes_ == 1)[0])
            scores = classifier.predict_proba(z_val)[:, fake_column]
            inference_sec = time.perf_counter() - started
            frame = score_frame(val, scores)
            report = evaluate_binary_predictions(frame)
            row = {
                "run_id": run_id,
                "holdout_generator": target,
                "model": "Frozen MERT + LR",
                "candidate_order": order,
                "C": c,
                "gamma": None,
                "layer": layer,
                "train_segments": len(train),
                "val_segments": len(val),
                "val_segment_auc": report["segment"]["roc_auc"],
                "val_segment_eer": report["segment"]["eer"],
                "track_roc_auc": report["track"]["roc_auc"],
                "track_eer": report["track"]["eer"],
                "val_track_macro_f1": report["track"]["macro_f1"],
                "fit_seconds": fit_sec,
                "val_inference_seconds": inference_sec,
                "warnings": json.dumps(
                    [str(item.message) for item in captured], ensure_ascii=False
                ),
            }
            rows.append(row)
            if select_row(rows)["candidate_order"] == order:
                payload = {
                    "run_id": run_id,
                    "holdout_generator": target,
                    "model": "m-a-p/MERT-v1-95M",
                    "revision": MERT_REVISION,
                    "layer": layer,
                    "C": c,
                    "scaler": scaler,
                    "classifier": classifier,
                    "feature_dim": 768,
                }
                joblib.dump(payload, checkpoint)
                selected = {
                    "row": row.copy(),
                    "thresholds": report["thresholds"],
                    "checkpoint": str(checkpoint),
                    "validation_frame": frame,
                    "validation_report": report,
                }
            log(
                f"{target} MERT layer {layer:02d}/12 C={c:g} "
                f"Val Track EER={row['track_eer']:.6f} AUC={row['track_roc_auc']:.6f}"
            )
    if (
        selected is None
        or select_row(rows)["candidate_order"] != selected["row"]["candidate_order"]
    ):
        raise RuntimeError("MERT selection does not match the full Validation table")
    save_validation_scores(
        selected["validation_report"],
        selected["validation_frame"],
        target,
        "Frozen MERT + LR",
        run_id,
        output,
    )
    selected["checkpoint_sha256"] = file_sha256(checkpoint)
    del selected["validation_frame"], selected["validation_report"]
    return rows, selected


def read_test_features(test: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Open Test handcrafted inputs only after all holdout selection is frozen."""
    chunks = []
    for chunk in pd.read_csv(FEATURE_PATH, chunksize=3000):
        chunks.append(chunk.loc[chunk["split"].eq("test")].copy())
    features = pd.concat(chunks, ignore_index=True)
    if features["segment_id"].duplicated().any():
        raise ValueError("Test feature file contains duplicate IDs")
    matched = test[["segment_id"]].merge(
        features, on="segment_id", validate="one_to_one"
    )
    if len(matched) != len(test):
        raise ValueError("Test feature file lacks a requested segment")
    matrix = matched[feature_columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(matrix.to_numpy(dtype=np.float64)).all():
        raise ValueError("Test handcrafted features contain invalid values")
    return matrix


def test_subset(manifest: pd.DataFrame, target: str) -> pd.DataFrame:
    """Use all original Test REAL and only the heldout generator's Test FAKE."""
    subset = (
        manifest.loc[
            manifest["split"].eq("test")
            & (manifest["label_id"].eq(0) | manifest["generator"].eq(target))
        ]
        .copy()
        .reset_index(drop=True)
    )
    if subset["label_id"].nunique() != 2 or subset["original_audio"].nunique() != 45:
        raise ValueError("Primary Test does not retain all 45 original REAL groups")
    if subset.loc[subset["label_id"].eq(0), "track_sample_id"].nunique() != 45:
        raise ValueError("Primary Test REAL track count differs")
    expected_fake = 45 if target == "musicgen" else 48
    if (
        subset.loc[subset["label_id"].eq(1), "track_sample_id"].nunique()
        != expected_fake
    ):
        raise ValueError("Primary Test target FAKE count differs")
    return subset


def paired_subset(primary: pd.DataFrame) -> pd.DataFrame:
    """Keep only source groups with target FAKE for a secondary sensitivity check."""
    fake_groups = set(primary.loc[primary["label_id"].eq(1), "original_audio"])
    return (
        primary.loc[primary["original_audio"].isin(fake_groups)]
        .copy()
        .reset_index(drop=True)
    )


def score_frozen_model(
    model_name: str,
    checkpoint: Path,
    test: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    mert_values: np.ndarray,
) -> tuple[pd.DataFrame, float]:
    """Run one selected model without changing its Train fit or Val thresholds."""
    started = time.perf_counter()
    if model_name == "LogisticRegression":
        pipeline = joblib.load(checkpoint)
        scores = score_lr(pipeline, feature_matrix)
    elif model_name == "RBF-SVM":
        pipeline = joblib.load(checkpoint)
        classifier = pipeline.named_steps["classifier"]
        if classifier.probability or not np.array_equal(classifier.classes_, [0, 1]):
            raise ValueError("SVM checkpoint is not a FAKE-direction margin model")
        scores = pipeline.decision_function(feature_matrix)
    else:
        payload = joblib.load(checkpoint)
        classifier = payload["classifier"]
        if not np.array_equal(classifier.classes_, [0, 1]):
            raise ValueError("MERT LR checkpoint class order differs")
        layer_values = np.asarray(
            mert_values[:, int(payload["layer"]), :], dtype=np.float32
        )
        fake_column = int(np.flatnonzero(classifier.classes_ == 1)[0])
        scores = classifier.predict_proba(payload["scaler"].transform(layer_values))[
            :, fake_column
        ]
    elapsed = time.perf_counter() - started
    return score_frame(test, scores), elapsed


def rows_from_report(
    report: dict,
    run_id: str,
    target: str,
    model_name: str,
    cohort: str,
    split: str,
) -> list[dict]:
    """Flatten the common evaluator's separate Segment and Track metrics."""
    return [
        {
            "run_id": run_id,
            "holdout_generator": target,
            "model": model_name,
            "variant": "optimized",
            "cohort": cohort,
            "split": split,
            "level": level,
            "validation_segment_threshold": report["thresholds"]["segment"],
            "validation_track_threshold": report["thresholds"]["track"],
            **report[level],
        }
        for level in ("segment", "track")
    ]


def main() -> None:
    """Run both strict holdouts, save choices, then evaluate their fixed Test sets."""
    random.seed(SEED)
    np.random.seed(SEED)
    run_id = "unseen_classical_mert_" + datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    output = RESULT_ROOT / run_id
    checkpoint_dir = CHECKPOINT_ROOT / run_id
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    protocol_path = ROOT / "docs/ROBUSTNESS_PROTOCOL.md"
    protocol_hash = file_sha256(protocol_path)
    if (
        protocol_hash
        != "95f1546ce8e3fcedb66e5ef52488c9255e414965514b5163e1a62b9052e4941e"
    ):
        raise ValueError("Predeclared robustness protocol changed before unseen Test")
    manifest, manifest_hash = read_manifest()
    feature_columns = json.loads(
        (ROOT / "results/model_tuning/classical_run_metadata.json").read_text(
            encoding="utf-8"
        )
    )["feature_columns"]
    features = read_training_features(manifest, feature_columns)
    train_val = manifest.loc[manifest["split"].isin(("train", "val"))].reset_index(
        drop=True
    )
    embeddings, cache_meta = read_mert_cache(MERT_TRAIN_STEM, train_val, manifest_hash)
    log(
        f"run_id={run_id} manifest={manifest_hash} Train/Val MERT cache={embeddings.shape}"
    )

    all_tuning: list[dict] = []
    selected: dict[str, dict] = {}
    inclusion: list[dict] = []
    model_order = ("LogisticRegression", "RBF-SVM", "Frozen MERT + LR")
    for target in TARGETS:
        train = filtered_split(train_val, target, "train")
        val = filtered_split(train_val, target, "val")
        for split_name, part in (("train", train), ("val", val)):
            for label_id, label in ((0, "REAL"), (1, "FAKE")):
                subset = part.loc[part["label_id"].eq(label_id)]
                inclusion.append(
                    {
                        "run_id": run_id,
                        "holdout_generator": target,
                        "split": split_name,
                        "label": label,
                        "original_audio": subset["original_audio"].nunique(),
                        "tracks": subset["track_sample_id"].nunique(),
                        "segments": len(subset),
                    }
                )
        # Every selection, including C, gamma, layer, and both thresholds,
        # is recomputed without the target generator in Train or Validation.
        selected[target] = {}
        for model_name, stem in (("LogisticRegression", "lr"), ("RBF-SVM", "svm")):
            path = checkpoint_dir / f"{target}_{stem}_best.joblib"
            rows, choice = train_classical(
                target,
                model_name,
                train,
                val,
                features,
                feature_columns,
                path,
                run_id,
                output,
            )
            all_tuning.extend(rows)
            selected[target][model_name] = choice
            pd.DataFrame(all_tuning).to_csv(output / "tuning_progress.csv", index=False)
        path = checkpoint_dir / f"{target}_mert_best.joblib"
        rows, choice = train_mert(
            target, train, val, train_val, embeddings, path, run_id, output
        )
        all_tuning.extend(rows)
        selected[target]["Frozen MERT + LR"] = choice
        pd.DataFrame(all_tuning).to_csv(output / "tuning_progress.csv", index=False)

    # 후보 수를 확인하고 선택 파일을 저장한 뒤에만 Test를 연다.
    tuning = pd.DataFrame(all_tuning)
    if len(tuning) != 2 * (
        len(LR_C) + len(SVM_C) * len(SVM_GAMMA) + len(MERT_LAYERS) * len(LR_C)
    ):
        raise RuntimeError("One or more holdout search candidates are missing")
    tuning.to_csv(output / "tuning.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(inclusion).to_csv(
        output / "inclusion_counts.csv", index=False, encoding="utf-8-sig"
    )
    selection_json = {
        target: {
            model: {key: value for key, value in choice.items() if key != "row"}
            | {"candidate": choice["row"]}
            for model, choice in selected[target].items()
        }
        for target in TARGETS
    }
    (output / "selected_configs.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "manifest_sha256": manifest_hash,
                "protocol_sha256": protocol_hash,
                "frozen_mert_revision": MERT_REVISION,
                "selection": "filtered Validation Track EER min, Track ROC-AUC max, candidate order",
                "candidate_counts_per_holdout": {
                    "LogisticRegression": 5,
                    "RBF-SVM": 16,
                    "Frozen MERT + LR": 65,
                },
                "models": selection_json,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    log(
        f"All 172 candidates selected after {time.perf_counter()-started:.1f}s; opening Test now"
    )

    # The Test feature table and Test MERT cache are first opened after both
    # holdout searches are frozen. All evaluation rows stay in original Test.
    # 먼저 공통 Test MERT cache의 전체 ID를 읽고 대상별 primary 집합은 뒤에서 추린다.
    test_all = manifest.loc[manifest["split"].eq("test")].reset_index(drop=True)
    test_embeddings, test_cache_meta = read_mert_cache(
        MERT_TEST_STEM, test_all, manifest_hash
    )
    test_positions = pd.Index(test_all["segment_id"])
    metrics: list[dict] = []
    paired_metrics: list[dict] = []
    raw_rows: list[pd.DataFrame] = []
    timings: list[dict] = []
    test_cohorts: list[dict] = []
    for target in TARGETS:
        primary = test_subset(manifest, target)
        paired = paired_subset(primary)
        held_fake_groups = primary.loc[
            primary["label_id"].eq(1), "original_audio"
        ].nunique()
        test_cohorts.append(
            {
                "run_id": run_id,
                "holdout_generator": target,
                "primary_real_tracks": primary.loc[
                    primary.label_id.eq(0), "track_sample_id"
                ].nunique(),
                "primary_fake_tracks": primary.loc[
                    primary.label_id.eq(1), "track_sample_id"
                ].nunique(),
                "primary_fake_original_audio": held_fake_groups,
                "primary_segments": len(primary),
                "paired_real_tracks": paired.loc[
                    paired.label_id.eq(0), "track_sample_id"
                ].nunique(),
                "paired_fake_tracks": paired.loc[
                    paired.label_id.eq(1), "track_sample_id"
                ].nunique(),
            }
        )
        feature_matrix = read_test_features(primary, feature_columns)
        positions = test_positions.get_indexer(primary["segment_id"])
        if (positions < 0).any():
            raise ValueError("MERT Test cache misses a primary cohort segment")
        mert_values = test_embeddings[positions]
        for model_name in model_order:
            choice = selected[target][model_name]
            checkpoint = Path(choice["checkpoint"])
            if file_sha256(checkpoint) != choice["checkpoint_sha256"]:
                raise ValueError("A frozen holdout checkpoint changed after selection")
            frame, inference_sec = score_frozen_model(
                model_name, checkpoint, primary, feature_matrix, mert_values
            )
            report = evaluate_binary_predictions(frame, thresholds=choice["thresholds"])
            metrics.extend(
                rows_from_report(report, run_id, target, model_name, "primary", "test")
            )
            for level in ("segment", "track"):
                raw = report[f"{level}_scores"].copy()
                for key, value in (
                    ("run_id", run_id),
                    ("holdout_generator", target),
                    ("model", model_name),
                    ("variant", "optimized"),
                    ("cohort", "primary"),
                    ("split", "test"),
                    ("level", level),
                    ("threshold", choice["thresholds"][level]),
                ):
                    raw[key] = value
                raw_rows.append(raw)
            paired_ids = set(paired["segment_id"])
            paired_frame = frame.loc[frame["segment_id"].isin(paired_ids)].copy()
            paired_report = evaluate_binary_predictions(
                paired_frame, thresholds=choice["thresholds"]
            )
            paired_metrics.extend(
                rows_from_report(
                    paired_report, run_id, target, model_name, "paired_source", "test"
                )
            )
            timings.append(
                {
                    "run_id": run_id,
                    "holdout_generator": target,
                    "model": model_name,
                    "inference_seconds": inference_sec,
                    "n_test_segments": len(primary),
                    "device": "CPU",
                    "excluded_mert_encoder_seconds": True,
                }
            )
            log(
                f"Test {target} {model_name}: Track AUC={report['track']['roc_auc']:.6f} "
                f"EER={report['track']['eer']:.6f} BA={report['track']['balanced_accuracy']:.6f}"
            )

    pd.DataFrame(metrics).to_csv(
        output / "primary_test_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(paired_metrics).to_csv(
        output / "paired_source_test_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(raw_rows, ignore_index=True).to_csv(
        output / "primary_test_scores.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(timings).to_csv(
        output / "test_inference_times.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(test_cohorts).to_csv(
        output / "test_cohorts.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "run_id": run_id,
        "manifest_sha256": manifest_hash,
        "protocol_sha256": protocol_hash,
        "frozen_selection_artifacts": {
            "tuning_sha256": file_sha256(output / "tuning.csv"),
            "selected_configs_sha256": file_sha256(output / "selected_configs.json"),
            "checkpoint_sha256": {
                target: {
                    model: selected[target][model]["checkpoint_sha256"]
                    for model in model_order
                }
                for target in TARGETS
            },
        },
        "targets": TARGETS,
        "seed": SEED,
        "selection_used_test": False,
        "search_reused_full_data_hyperparameters": False,
        "historical_test_exposure": (
            "MusicGen/Udio and the original Test were examined in 2026-09-13 "
            "historical experiments; this is not an untouched holdout."
        ),
        "cohort_definition": (
            "All original Test REAL plus target-generator Test FAKE; paired-source "
            "subset is secondary and never used for selection."
        ),
        "test_cohorts": test_cohorts,
        "mert_train_val_cache": cache_meta,
        "mert_test_cache": test_cache_meta,
        "feature_columns_sha256": hashlib.sha256(
            "\n".join(feature_columns).encode()
        ).hexdigest(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "device": "CPU",
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "run_metadata.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (RESULT_ROOT / "latest_run.json").write_text(
        json.dumps({"run_id": run_id, "result_dir": str(output)}, indent=2),
        encoding="utf-8",
    )
    log(f"Completed {run_id} in {summary['elapsed_seconds']:.1f}s; outputs: {output}")


if __name__ == "__main__":
    main()
