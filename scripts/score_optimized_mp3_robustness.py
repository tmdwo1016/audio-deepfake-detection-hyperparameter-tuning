"""Score four frozen optimized models on paired full-decode clean/MP3 Test audio.

Requires caches from extract_paired_mp3_features.py and extract_mert_mp3.py.
Uses only the existing clean Validation thresholds; never selects settings on Test.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", tempfile.gettempdir())
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.cnn_modeling import LogMelCNN, LogMelDataset, predict_scores
from src.modeling_final import evaluate_frozen_test_run, verify_test_score_frame
from src.modeling_final_run import load_fixed_test, preflight, score_frame
from src.mp3_robustness_inputs import (
    ALL_CONDITIONS,
    fixed_test_manifest,
    load_track,
    sha256_file,
    slice_segment,
    validate_paired_mp3,
)
from scripts.extract_paired_mp3_features import logmel_features

MODELS = ("LogisticRegression", "RBF-SVM", "Log-Mel CNN", "Frozen MERT + LR")
PROTOCOL_SHA256 = "95f1546ce8e3fcedb66e5ef52488c9255e414965514b5163e1a62b9052e4941e"


def source_metadata(root: Path, context: dict, test: pd.DataFrame) -> dict:
    """Bind this experiment to the completed 23 run and predeclared protocol."""
    protocol_path = root / "docs/ROBUSTNESS_PROTOCOL.md"
    # 사전 고정한 MP3 조건이 바뀌었으면 Test 점수 계산을 시작하지 않는다.
    if sha256_file(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("The predeclared robustness protocol changed")
    final = json.loads(
        (root / "results/model_tuning/final_run_metadata.json").read_text()
    )
    if final["split_manifest_sha256"] != context["manifest_hash"]:
        raise ValueError("Source final Test manifest differs")
    if final["component_runs"] != context["run_ids"]:
        raise ValueError("Source final Test components differ")
    _, report_hash = validate_paired_mp3(root, test)
    return {
        "source_final_run_id": final["final_run_id"],
        "component_runs": final["component_runs"],
        "split_manifest_sha256": context["manifest_hash"],
        "transcode_report_sha256": report_hash,
        "protocol_sha256": PROTOCOL_SHA256,
        "conditions": list(ALL_CONDITIONS),
        "bitrate_kbps": {"mp3_128": 128, "mp3_64": 64},
        "decode": "full track from beginning, 24kHz mono, original segment start_sec",
        "thresholds": "frozen clean Validation segment/track values from final 23 run",
    }


def validate_array_cache(
    root: Path, condition: str, kind: str, shape: tuple[int, ...], provenance: dict
) -> Path:
    """Require a complete cache bound to the same IDs and protocol."""
    directory = root / "data/processed/model_robustness/mp3" / kind
    if kind == "handcrafted":
        data_path = directory / f"{condition}_features.npy"
        done_path = directory / f"{condition}_done.npy"
        meta_path = directory / f"{condition}_meta.json"
    elif kind == "logmel" and condition == "clean":
        data_path = directory / "clean_float16.npy"
        done_path = directory / "clean_done.npy"
        meta_path = directory / "clean_meta.json"
    else:
        raise ValueError("Only paired handcrafted and clean Log-Mel caches are local")
    for path in (data_path, done_path, meta_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    for key in (
        "condition",
        "manifest_sha256",
        "transcode_report_sha256",
        "protocol_sha256",
    ):
        if metadata[key] != provenance[key]:
            raise ValueError(f"{condition} {kind} cache differs at {key}")
    array = np.load(data_path, mmap_mode="r")
    done = np.load(done_path)
    if array.shape != shape or done.shape != (shape[0],) or not bool(done.all()):
        raise ValueError(f"Incomplete {condition} {kind} cache")
    return data_path


def mert_cache(
    root: Path, condition: str, test: pd.DataFrame, provenance: dict, revision: str
) -> tuple[Path, dict]:
    directory = root / "data/processed/model_robustness/mp3/mert"
    stem = f"mert95m_{condition}_validframe_{revision[:12]}"
    data_path = directory / f"{stem}.npy"
    done_path = directory / f"{stem}_done.npy"
    index_path = directory / f"{stem}_index.csv"
    meta_path = directory / f"{stem}_meta.json"
    run_path = directory / f"{stem}_run.json"
    for path in (data_path, done_path, index_path, meta_path, run_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if (
        metadata["revision"] != revision
        or metadata["manifest_sha256"] != provenance["split_manifest_sha256"]
        or metadata["transcode_report_sha256"] != provenance["transcode_report_sha256"]
        or metadata["condition"] != condition
    ):
        raise ValueError(f"MERT {condition} provenance differs")
    index = pd.read_csv(index_path)
    if (
        index["segment_id"].astype(str).tolist()
        != test["segment_id"].astype(str).tolist()
    ):
        raise ValueError(f"MERT {condition} segment order differs")
    data = np.load(data_path, mmap_mode="r")
    done = np.load(done_path)
    if (
        data.shape != (len(test), 13, 768)
        or done.shape != (len(test),)
        or not bool(done.all())
    ):
        raise ValueError(f"MERT {condition} cache incomplete")
    summary = json.loads(run_path.read_text(encoding="utf-8"))
    if summary["metadata"] != metadata:
        raise ValueError(f"MERT {condition} run summary differs")
    return data_path, summary


def score_classical(context: dict, test: pd.DataFrame, features: Path) -> dict:
    """Apply the two frozen Train-only Pipeline checkpoints to paired features."""
    columns = context["classical_meta"]["feature_columns"]
    values = np.load(features, mmap_mode="r")
    if values.shape != (len(test), len(columns)) or not np.isfinite(values).all():
        raise ValueError("Paired 266-D features are invalid")
    matrix = pd.DataFrame(np.asarray(values), columns=columns)
    scored = {}
    for model, stem in (("LogisticRegression", "logistic"), ("RBF-SVM", "svm")):
        pipeline = joblib.load(context["checkpoints"] / f"{stem}_best.joblib")
        if list(pipeline.feature_names_in_) != columns:
            raise ValueError(f"{model} feature order differs")
        classifier = pipeline.named_steps["classifier"]
        if not np.array_equal(classifier.classes_, [0, 1]):
            raise ValueError(f"{model} FAKE class index differs")
        started = time.perf_counter()
        scores = (
            pipeline.predict_proba(matrix)[:, 1]
            if model == "LogisticRegression"
            else pipeline.decision_function(matrix)
        )
        scored[model] = (score_frame(test, scores), time.perf_counter() - started)
    return scored


def score_cnn(
    context: dict, test: pd.DataFrame, cache_path: Path
) -> tuple[pd.DataFrame, float, str]:
    """Restore optimized CNN and score a manifest-ordered Log-Mel cache."""
    checkpoint = context["cnn_checkpoint"]["optimized"]
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else (
            "mps"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
    )
    model = LogMelCNN(dropout=float(checkpoint["config"]["dropout"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    index = pd.DataFrame(
        {
            "segment_id": test["segment_id"],
            "logmel_index": np.arange(len(test), dtype=int),
        }
    )
    dataset = LogMelDataset(test, cache_path, index)
    loader = DataLoader(
        dataset,
        batch_size=int(checkpoint["config"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    started = time.perf_counter()
    scored = predict_scores(model, loader, test, device)
    return scored, time.perf_counter() - started, str(device)


def verify_historical_logmel(
    root: Path, test: pd.DataFrame, condition: str, cache_path: Path
) -> None:
    """Spot-check legacy full-decode cache order and transform against MP3 audio."""
    cached = np.load(cache_path, mmap_mode="r")
    for index in (0, len(test) // 2, len(test) - 1):
        row = test.iloc[index]
        waveform = load_track(root, condition, str(row["track_sample_id"]))
        expected = logmel_features(slice_segment(waveform, row["start_sec"], pad=True))
        if not np.array_equal(cached[index], expected):
            raise ValueError(
                f"{condition} full-decode Log-Mel cache order/values differ at {row['segment_id']}"
            )


def score_mert(
    context: dict, test: pd.DataFrame, cache_path: Path
) -> tuple[pd.DataFrame, float]:
    """Apply the frozen optimized scaler and LR head to one selected MERT layer."""
    payload = context["mert_checkpoint"]["optimized"]
    values = np.asarray(
        np.load(cache_path, mmap_mode="r")[:, int(payload["layer"]), :],
        dtype=np.float32,
    )
    if not np.isfinite(values).all() or values.shape != (len(test), 768):
        raise ValueError("Paired MERT embedding is invalid")
    classifier = payload["classifier"]
    if not np.array_equal(classifier.classes_, [0, 1]):
        raise ValueError("MERT FAKE class index differs")
    started = time.perf_counter()
    scaled = payload["scaler"].transform(values)
    scores = classifier.predict_proba(scaled)[:, 1]
    return score_frame(test, scores), time.perf_counter() - started


def thresholds_for(context: dict, model: str) -> dict[str, float]:
    if model in ("LogisticRegression", "RBF-SVM"):
        return context["classical_thresholds"]["models"][model]["optimized"]
    if model == "Log-Mel CNN":
        return context["cnn_selection"]["optimized"]["thresholds"]
    return context["mert_checkpoint"]["optimized"]["thresholds"]


def historical_clean_scores(
    root: Path, model: str, test: pd.DataFrame, expected_run_id: str
) -> pd.DataFrame:
    stem = model.lower().replace(" ", "_").replace("-", "_")
    path = (
        root
        / "results/model_tuning/test_scores"
        / f"{stem}_optimized_test_segment_scores.csv"
    )
    raw = pd.read_csv(path)
    if (
        set(raw["run_id"]) != {expected_run_id}
        or set(raw["split"]) != {"test"}
        or set(raw["level"]) != {"segment"}
    ):
        raise ValueError(f"23 clean score provenance differs for {model}")
    return verify_test_score_frame(test, raw)


def final_reference_table(root: Path, model: str, level: str) -> pd.Series:
    table = pd.read_csv(root / "results/model_tuning/optimized_test_results.csv")
    rows = table.loc[
        (table["model"] == model)
        & (table["variant"] == "optimized")
        & (table["level"] == level)
    ]
    if len(rows) != 1:
        raise ValueError("Source 23 optimized metric row missing")
    return rows.iloc[0]


def save_plots(metrics: pd.DataFrame, output: Path) -> None:
    """Plot paired Track AUC, EER and fixed-threshold HTER by bitrate."""
    track = metrics.loc[metrics["level"].eq("track")]
    for measure, title in (
        ("roc_auc", "Track ROC-AUC"),
        ("eer", "Track EER"),
        ("hter", "Track HTER at clean Validation threshold"),
    ):
        pivot = track.pivot(index="model", columns="condition", values=measure).loc[
            list(MODELS), list(ALL_CONDITIONS)
        ]
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(MODELS))
        for offset, condition in zip((-0.25, 0, 0.25), ALL_CONDITIONS, strict=True):
            ax.bar(x + offset, pivot[condition], width=0.24, label=condition)
        ax.set_xticks(x, MODELS, rotation=15, ha="right")
        ax.set_ylabel(measure)
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / f"paired_track_{measure}.png", dpi=160)
        plt.close(fig)


# 입력: 23번의 원본 clean 표현과 새 전체 디코딩 clean 표현.
# 출력: 표현별 최대·평균 절댓값 차이와 달라진 Segment 수.
def compare_clean_inputs(
    root: Path,
    context: dict,
    test: pd.DataFrame,
    original_features: pd.DataFrame,
    paired_features: Path,
    paired_logmel: Path,
    paired_mert: Path,
) -> dict:
    """Measure clean extraction differences from final 23 before paired scoring."""
    original = original_features.to_numpy(dtype=np.float64)
    paired = np.asarray(np.load(paired_features, mmap_mode="r"))
    feature_diff = np.abs(original - paired)
    index = pd.read_csv(context["logmel_index_path"])
    positions = (
        index.set_index("segment_id")
        .loc[test["segment_id"], "logmel_index"]
        .to_numpy(dtype=int)
    )
    original_mel = np.load(context["logmel_path"], mmap_mode="r")
    paired_mel = np.load(paired_logmel, mmap_mode="r")
    mel_max = 0.0
    mel_sum = 0.0
    mel_count = 0
    mel_different_segments = 0
    for row, position in enumerate(positions):
        difference = np.abs(
            np.asarray(original_mel[position], dtype=np.float32)
            - np.asarray(paired_mel[row], dtype=np.float32)
        )
        mel_max = max(mel_max, float(difference.max()))
        mel_sum += float(difference.sum())
        mel_count += difference.size
        mel_different_segments += bool(np.any(difference > 0))
    final = json.loads(
        (root / "results/model_tuning/final_run_metadata.json").read_text()
    )
    original_mert_path = Path(final["mert"]["cache_paths"]["embeddings"])
    layer = int(context["mert_checkpoint"]["optimized"]["layer"])
    original_mert = np.asarray(
        np.load(original_mert_path, mmap_mode="r")[:, layer, :], dtype=np.float32
    )
    paired_mert_values = np.asarray(
        np.load(paired_mert, mmap_mode="r")[:, layer, :], dtype=np.float32
    )
    mert_diff = np.abs(original_mert - paired_mert_values)
    return {
        "comparison": "23 offset-loaded clean inputs vs new full-decode paired clean inputs",
        "handcrafted": {
            "max_abs_diff": float(feature_diff.max()),
            "mean_abs_diff": float(feature_diff.mean()),
            "segments_with_any_diff_gt_1e_6": int(
                np.sum(np.any(feature_diff > 1e-6, axis=1))
            ),
        },
        "logmel": {
            "max_abs_diff": mel_max,
            "mean_abs_diff": mel_sum / mel_count,
            "segments_with_any_exact_diff": mel_different_segments,
        },
        "mert_selected_layer": {
            "layer": layer,
            "max_abs_diff": float(mert_diff.max()),
            "mean_abs_diff": float(mert_diff.mean()),
            "segments_with_any_exact_diff": int(np.sum(np.any(mert_diff > 0, axis=1))),
        },
    }


# 입력: 세 조건의 검증된 cache와 기존 Optimized checkpoint/임계값.
# 출력: 조건별 원점수·지표·paired 변화량·출처 파일을 새 실행 폴더에 저장.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    context = preflight(root)
    test, original_features, _ = load_fixed_test(context)
    fixed, _ = fixed_test_manifest(root)
    if test["segment_id"].tolist() != fixed["segment_id"].tolist():
        raise ValueError("Fixed Test ordering differs")
    provenance = source_metadata(root, context, test)
    cache_provenance = {
        **provenance,
        "manifest_sha256": provenance["split_manifest_sha256"],
    }
    run_id = args.run_id or "mp3_optimized_" + datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    output = root / "results/model_robustness/mp3" / run_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Robustness run output already exists: {output}")

    # 세 조건의 표현 cache가 모두 완성되고 같은 manifest를 쓰는지 먼저 확인한다.
    feature_paths = {
        condition: validate_array_cache(
            root,
            condition,
            "handcrafted",
            (len(test), 266),
            {**cache_provenance, "condition": condition},
        )
        for condition in ALL_CONDITIONS
    }
    logmel_paths = {
        "clean": validate_array_cache(
            root,
            "clean",
            "logmel",
            (len(test), 128, 1001),
            {**cache_provenance, "condition": "clean"},
        )
    }
    for condition in ("mp3_128", "mp3_64"):
        path = (
            root
            / "data/processed/logmel/mp3_robustness"
            / f"logmel_test_{condition}_fulldecode_float16.npy"
        )
        done = path.with_name(path.stem.replace("_float16", "_done") + ".npy")
        if np.load(path, mmap_mode="r").shape != (len(test), 128, 1001) or not bool(
            np.load(done).all()
        ):
            raise ValueError(f"Historical full-decode CNN {condition} cache incomplete")
        verify_historical_logmel(root, test, condition, path)
        logmel_paths[condition] = path
    revision = context["mert_checkpoint"]["optimized"]["revision"]
    mert_paths, mert_summaries = {}, {}
    for condition in ALL_CONDITIONS:
        mert_paths[condition], mert_summaries[condition] = mert_cache(
            root,
            condition,
            test,
            provenance,
            revision,
        )

    # 새 paired clean과 23번 clean의 전처리 차이를 별도 품질 기록으로 남긴다.
    input_qc = compare_clean_inputs(
        root,
        context,
        test,
        original_features,
        feature_paths["clean"],
        logmel_paths["clean"],
        mert_paths["clean"],
    )

    # No output is created until all three paired input conditions pass preflight.
    output.mkdir(parents=True, exist_ok=True)
    (output / "clean_input_reference_qc.json").write_text(
        json.dumps(input_qc, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metrics_rows, timing_rows, scored_by_model = [], [], {model: {} for model in MODELS}
    for condition in ALL_CONDITIONS:
        print("Scoring", condition, flush=True)
        classical = score_classical(context, test, feature_paths[condition])
        cnn_frame, cnn_seconds, cnn_device = score_cnn(
            context, test, logmel_paths[condition]
        )
        mert_frame, mert_seconds = score_mert(context, test, mert_paths[condition])
        condition_scores = {
            **classical,
            "Log-Mel CNN": (cnn_frame, cnn_seconds),
            "Frozen MERT + LR": (mert_frame, mert_seconds),
        }
        for model in MODELS:
            frame, seconds = condition_scores[model]
            scored_by_model[model][condition] = frame
            source = (
                context["run_ids"]["classical"]
                if model in MODELS[:2]
                else (
                    context["run_ids"]["cnn"]
                    if model == "Log-Mel CNN"
                    else context["run_ids"]["mert"]
                )
            )
            rows = evaluate_frozen_test_run(
                test,
                frame,
                thresholds_for(context, model),
                model=model,
                variant="optimized",
                run_id=source,
                output_dir=output / "raw_scores" / condition,
            )
            stem = model.lower().replace(" ", "_").replace("-", "_")
            for level in ("segment", "track"):
                path = (
                    output
                    / "raw_scores"
                    / condition
                    / f"{stem}_optimized_test_{level}_scores.csv"
                )
                raw = pd.read_csv(path)
                raw.insert(0, "condition", condition)
                raw.insert(0, "robustness_run_id", run_id)
                raw.to_csv(path, index=False, encoding="utf-8-sig")
            for row in rows:
                metrics_rows.append(
                    {
                        "robustness_run_id": run_id,
                        "condition": condition,
                        "source_final_run_id": provenance["source_final_run_id"],
                        **row,
                    }
                )
            timing_rows.append(
                {
                    "robustness_run_id": run_id,
                    "condition": condition,
                    "model": model,
                    "inference_seconds": seconds,
                    "device": cnn_device if model == "Log-Mel CNN" else "cpu",
                }
            )
        print("Scored", condition, flush=True)

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(output / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(timing_rows).to_csv(output / "inference_times.csv", index=False)
    measures = (
        "roc_auc",
        "ap_fake",
        "ap_real",
        "eer",
        "balanced_accuracy",
        "macro_f1",
        "real_fpr",
        "fake_miss_rate",
        "hter",
    )
    comparisons = []
    for model in MODELS:
        for level in ("segment", "track"):
            clean = metrics.loc[
                (metrics.model == model)
                & (metrics.level == level)
                & (metrics.condition == "clean")
            ].iloc[0]
            for condition in ("mp3_128", "mp3_64"):
                shifted = metrics.loc[
                    (metrics.model == model)
                    & (metrics.level == level)
                    & (metrics.condition == condition)
                ].iloc[0]
                row = {
                    "robustness_run_id": run_id,
                    "model": model,
                    "level": level,
                    "condition": condition,
                    "clean_n": int(clean["n"]),
                    "mp3_n": int(shifted["n"]),
                }
                for measure in measures:
                    row[f"clean_{measure}"] = float(clean[measure])
                    row[f"mp3_{measure}"] = float(shifted[measure])
                    row[f"delta_{measure}"] = float(shifted[measure] - clean[measure])
                comparisons.append(row)
    # 변화량은 같은 ID의 새 paired clean을 기준으로 계산해 저장한다.
    pd.DataFrame(comparisons).to_csv(
        output / "paired_deltas.csv", index=False, encoding="utf-8-sig"
    )

    paired_scores, reference = [], []
    for model in MODELS:
        frames = scored_by_model[model]
        for level in ("segment", "track"):
            id_column = "segment_id" if level == "segment" else "track_id"
            indexed = []
            for condition in ALL_CONDITIONS:
                frame = frames[condition]
                if level == "track":
                    frame = (
                        frame.groupby("track_id", sort=False)
                        .agg(
                            original_audio_id=("original_audio_id", "first"),
                            label=("label", "first"),
                            score=("score", "mean"),
                        )
                        .reset_index()
                    )
                indexed.append(
                    frame.set_index(id_column)["score"].rename(f"score_{condition}")
                )
            joined = pd.concat(indexed, axis=1)
            if joined.isna().any().any() or len(joined) != (
                1572 if level == "segment" else 539
            ):
                raise ValueError(f"Paired {model} {level} IDs differ")
            base = (
                frames["clean"][
                    ["segment_id", "track_id", "original_audio_id", "label"]
                ]
                if level == "segment"
                else frames["clean"]
                .groupby("track_id", sort=False)
                .agg(
                    original_audio_id=("original_audio_id", "first"),
                    label=("label", "first"),
                )
                .reset_index()
            )
            result = base.merge(
                joined.reset_index(), on=id_column, validate="one_to_one"
            )
            result.insert(0, "level", level)
            result.insert(0, "model", model)
            result.insert(0, "robustness_run_id", run_id)
            paired_scores.append(result)
        prior = historical_clean_scores(
            root,
            model,
            test,
            (
                context["run_ids"]["classical"]
                if model in MODELS[:2]
                else (
                    context["run_ids"]["cnn"]
                    if model == "Log-Mel CNN"
                    else context["run_ids"]["mert"]
                )
            ),
        )
        now = frames["clean"]
        joined = now[["segment_id", "score"]].merge(
            prior[["segment_id", "score"]],
            on="segment_id",
            suffixes=("_paired", "_23"),
            validate="one_to_one",
        )
        difference = np.abs(joined.score_paired - joined.score_23)
        for level in ("segment", "track"):
            current = metrics.loc[
                (metrics.model == model)
                & (metrics.level == level)
                & (metrics.condition == "clean")
            ].iloc[0]
            original = final_reference_table(root, model, level)
            reference.append(
                {
                    "model": model,
                    "level": level,
                    "segment_score_max_abs_diff": float(difference.max()),
                    "segment_score_mean_abs_diff": float(difference.mean()),
                    **{
                        f"delta_{measure}": float(current[measure] - original[measure])
                        for measure in measures
                    },
                }
            )
    pd.concat(paired_scores, ignore_index=True).to_csv(
        output / "paired_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(reference).to_csv(
        output / "clean_vs_final23.csv", index=False, encoding="utf-8-sig"
    )
    save_plots(metrics, output)
    provenance.update(
        {
            "robustness_run_id": run_id,
            "test_segments": len(test),
            "test_tracks": test.track_sample_id.nunique(),
            "class_counts_segment": test.label_id.value_counts().sort_index().to_dict(),
            "model_checkpoints": {
                "LogisticRegression": sha256_file(
                    context["checkpoints"] / "logistic_best.joblib"
                ),
                "RBF-SVM": sha256_file(context["checkpoints"] / "svm_best.joblib"),
                "Log-Mel CNN": sha256_file(context["checkpoints"] / "cnn_best.pt"),
                "Frozen MERT + LR": sha256_file(
                    context["checkpoints"] / "mert_best.joblib"
                ),
            },
            "mert_extraction": {
                condition: {
                    "model_load_seconds": summary["model_load_seconds"],
                    "embedding_extraction_seconds": summary[
                        "embedding_extraction_seconds"
                    ],
                    "device": summary["device"],
                    "cache_path": str(mert_paths[condition]),
                    "cache_sha256": sha256_file(mert_paths[condition]),
                    "cache_metadata_path": summary["paths"]["metadata"],
                    "cache_metadata_sha256": sha256_file(summary["paths"]["metadata"]),
                    "cache_provenance": {
                        "condition": summary["metadata"]["condition"],
                        "manifest_sha256": summary["metadata"]["manifest_sha256"],
                        "transcode_report_sha256": summary["metadata"][
                            "transcode_report_sha256"
                        ],
                        "model_revision": summary["metadata"]["revision"],
                        "decode": summary["metadata"]["decode"],
                        "padding": summary["metadata"]["padding"],
                    },
                }
                for condition, summary in mert_summaries.items()
            },
            "cnn_device": cnn_device,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
        }
    )
    # 프로토콜·모델·MERT cache 해시를 기록해 점수의 출처를 재검증할 수 있게 한다.
    (output / "run_metadata.json").write_text(
        json.dumps(
            provenance,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print("Paired MP3 optimized evaluation complete:", output)


if __name__ == "__main__":
    main()
