"""Run the frozen eight-variant Test evaluation after all Validation tuning.

The public notebook calls the stages separately so Test is read only after
preflight. Historical Test artifacts in this repository are kept separate.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", tempfile.gettempdir())

import torch
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from torch.utils.data import DataLoader

from src.cnn_modeling import LogMelCNN, LogMelDataset, predict_scores
from src.mert_modeling import (
    MODEL_NAME,
    MODEL_REVISION,
    embedding_cache_metadata,
)
from src.modeling_evaluation import assert_group_disjoint, select_best_candidate
from src.modeling_final import comparison_tables, evaluate_frozen_test_run


def file_hash(path: Path) -> str:
    """Hash exact file bytes for provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    """Read one frozen configuration or Validation threshold file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_thresholds(mapping: dict) -> bool:
    return set(mapping) == {"segment", "track"} and all(
        np.isfinite(float(value)) for value in mapping.values()
    )


# 입력: 프로젝트 경로. 출력: checkpoint·Validation 선택·입력 해시를 검증한 context.
# 이 함수가 끝나기 전에는 Test 특징과 점수를 읽지 않는다.
def preflight(project_root: str | Path) -> dict:
    """Require all eight checkpoints and saved Validation selections before Test."""
    root = Path(project_root).resolve()
    out = root / "results/model_tuning"
    checkpoints = root / "checkpoints/optimized"
    manifest_path = root / "data/metadata/segment_manifest_10s.csv"
    feature_path = root / "data/processed/features/handcrafted_features_10s.csv"
    logmel_path = root / "data/processed/logmel/logmel_10s_float16.npy"
    logmel_index_path = root / "data/processed/logmel/logmel_10s_index.csv"
    logmel_done_path = root / "data/processed/logmel/logmel_10s_done.npy"
    # 모든 후보 선택·checkpoint가 준비된 경우에만 Test를 읽을 수 있다.
    required = [
        manifest_path,
        feature_path,
        logmel_path,
        logmel_index_path,
        logmel_done_path,
        out / "classical_validation_thresholds.json",
        out / "classical_baseline_configs.json",
        out / "classical_best_configs.json",
        out / "classical_run_metadata.json",
        out / "classical_predeclared_configs.json",
        out / "logistic_tuning.csv",
        out / "svm_tuning.csv",
        out / "cnn_baseline_selection.json",
        out / "cnn_optimized_selection.json",
        out / "cnn_tuning.csv",
        out / "cnn_best_config.json",
        *[
            checkpoints / f"{stem}_{kind}.joblib"
            for stem in ("logistic", "svm")
            for kind in ("baseline", "best")
        ],
        *[checkpoints / f"cnn_{kind}.pt" for kind in ("baseline", "best")],
        *[checkpoints / f"mert_{kind}.joblib" for kind in ("baseline", "best")],
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "모든 Train/Validation 선택을 완료한 뒤 Test를 여세요:\n"
            + "\n".join(missing)
        )
    # 학습 산출물의 manifest 해시를 맞춰 다른 split에서 만든 모델 혼용을 막는다.
    manifest_hash = file_hash(manifest_path)
    index_hash = file_hash(logmel_index_path)
    classical_thresholds = read_json(out / "classical_validation_thresholds.json")
    classical_baselines = read_json(out / "classical_baseline_configs.json")
    classical_best = read_json(out / "classical_best_configs.json")
    classical_meta = read_json(out / "classical_run_metadata.json")
    classical_predeclared = read_json(out / "classical_predeclared_configs.json")
    classical_parts = [
        classical_thresholds,
        classical_baselines,
        classical_best,
        classical_meta,
    ]
    if any(part["split_manifest_sha256"] != manifest_hash for part in classical_parts):
        raise ValueError("Classical artifact manifest hash mismatch")
    if len({part["run_id"] for part in classical_parts}) != 1:
        raise ValueError("Classical run IDs disagree")
    if classical_predeclared["run_id"] != classical_meta["run_id"]:
        raise ValueError("Classical predeclared search run ID differs")
    for model, filename in (
        ("LogisticRegression", "logistic_tuning.csv"),
        ("RBF-SVM", "svm_tuning.csv"),
    ):
        tuning = pd.read_csv(out / filename).sort_values("candidate_order")
        candidates = classical_predeclared["candidates"][model]
        if (
            len(tuning) != len(candidates)
            or tuning["candidate_order"].tolist() != list(range(len(candidates)))
            or set(tuning["run_id"]) != {classical_meta["run_id"]}
            or set(tuning["model"]) != {model}
        ):
            raise ValueError(f"{model} Validation search is incomplete or mixed")
        for row, candidate in zip(tuning.to_dict("records"), candidates, strict=True):
            if not np.isclose(float(row["C"]), float(candidate["C"])):
                raise ValueError(f"{model} candidate C differs from predeclared search")
            if model == "RBF-SVM":
                expected_gamma = candidate["gamma"]
                gamma = row["gamma"]
                if (
                    gamma != expected_gamma
                    if expected_gamma == "scale"
                    else not np.isclose(float(gamma), float(expected_gamma))
                ):
                    raise ValueError(
                        "SVM candidate gamma differs from predeclared search"
                    )
        best_row = select_best_candidate(tuning)
        saved = classical_best["models"][model]
        if int(best_row["candidate_order"]) != int(
            saved["candidate_order"]
        ) or not np.isclose(float(best_row["C"]), float(saved["params"]["C"])):
            raise ValueError(f"{model} saved configuration is not Validation-best")
        if model == "RBF-SVM":
            saved_gamma = saved["params"]["gamma"]
            if (
                best_row["gamma"] != saved_gamma
                if saved_gamma == "scale"
                else not np.isclose(float(best_row["gamma"]), float(saved_gamma))
            ):
                raise ValueError("SVM saved gamma is not Validation-best")
        saved_thresholds = classical_thresholds["models"][model]["optimized"]
        for level in ("segment", "track"):
            if not np.isclose(
                float(saved_thresholds[level]),
                float(best_row[f"val_{level}_threshold"]),
            ):
                raise ValueError(f"{model} saved Validation {level} threshold differs")
        baseline = classical_baselines["models"][model]
        baseline_row = tuning.iloc[int(baseline["candidate_order"])]
        expected_baseline = classical_predeclared["baseline"][model]
        if not np.isclose(
            float(baseline_row["C"]), float(expected_baseline["C"])
        ) or not np.isclose(
            float(baseline["params"]["C"]), float(expected_baseline["C"])
        ):
            raise ValueError(f"{model} fixed baseline C differs")
        if model == "RBF-SVM" and (
            baseline_row["gamma"] != expected_baseline["gamma"]
            or baseline["params"]["gamma"] != expected_baseline["gamma"]
        ):
            raise ValueError("SVM fixed baseline gamma differs")
        for level in ("segment", "track"):
            if not np.isclose(
                float(classical_thresholds["models"][model]["baseline"][level]),
                float(baseline_row[f"val_{level}_threshold"]),
            ):
                raise ValueError(
                    f"{model} baseline Validation {level} threshold differs"
                )
    cnn_selection = {
        variant: read_json(out / f"cnn_{variant}_selection.json")
        for variant in ("baseline", "optimized")
    }
    cnn_checkpoint = {
        variant: torch.load(
            checkpoints / f"cnn_{'baseline' if variant == 'baseline' else 'best'}.pt",
            map_location="cpu",
            weights_only=False,
        )
        for variant in ("baseline", "optimized")
    }
    mert_checkpoint = {
        variant: joblib.load(
            checkpoints
            / f"mert_{'baseline' if variant == 'baseline' else 'best'}.joblib"
        )
        for variant in ("baseline", "optimized")
    }
    for variant in ("baseline", "optimized"):
        selection = cnn_selection[variant]
        checkpoint = cnn_checkpoint[variant]
        if (
            selection["manifest_sha256"] != manifest_hash
            or checkpoint["manifest_sha256"] != manifest_hash
        ):
            raise ValueError("CNN manifest hash mismatch")
        if (
            selection["cache_index_sha256"] != index_hash
            or checkpoint["cache_index_sha256"] != index_hash
        ):
            raise ValueError("CNN cache index hash mismatch")
        if (
            selection["run_id"] != checkpoint["run_id"]
            or selection["mel_settings"] != checkpoint["mel_settings"]
        ):
            raise ValueError("CNN checkpoint/selection mismatch")
        trial = selection["trial"]
        config = checkpoint["config"]
        if int(trial["best_epoch"]) != int(checkpoint["epoch"]):
            raise ValueError("CNN selected epoch/checkpoint mismatch")
        for key in (
            "lr",
            "dropout",
            "weight_decay",
            "batch_size",
            "max_epochs",
            "patience",
        ):
            if trial[key] != config[key]:
                raise ValueError(f"CNN selected {key}/checkpoint mismatch")
        if not np.isclose(
            float(trial["track_eer"]), float(checkpoint["val_track_eer"])
        ):
            raise ValueError("CNN selected EER/checkpoint mismatch")
        if not np.isclose(
            float(trial["track_roc_auc"]), float(checkpoint["val_track_roc_auc"])
        ):
            raise ValueError("CNN selected AUC/checkpoint mismatch")
        if not _finite_thresholds(selection["thresholds"]):
            raise ValueError("CNN Validation thresholds missing")
        payload = mert_checkpoint[variant]
        if payload["manifest_sha256"] != manifest_hash:
            raise ValueError("MERT manifest hash mismatch")
        if payload["model"] != MODEL_NAME or payload["revision"] != MODEL_REVISION:
            raise ValueError("MERT model/revision mismatch")
        if int(payload["feature_dim"]) != 768 or int(payload["layer"]) not in range(13):
            raise ValueError("MERT feature/layer mismatch")
        if not _finite_thresholds(payload["thresholds"]):
            raise ValueError("MERT Validation thresholds missing")
    if cnn_selection["baseline"]["run_id"] != cnn_selection["optimized"]["run_id"]:
        raise ValueError("CNN variants mix run IDs")
    cnn_tuning = pd.read_csv(out / "cnn_tuning.csv")
    if (
        len(cnn_tuning) != 18
        or cnn_tuning["trial_id"].duplicated().any()
        or cnn_tuning["candidate_order"].duplicated().any()
    ):
        raise ValueError(
            "CNN baseline plus 17 staged candidates are incomplete or duplicated"
        )
    stage_counts = cnn_tuning["stage"].value_counts().to_dict()
    if stage_counts != {"baseline": 1, "stage1": 9, "stage2": 8}:
        raise ValueError(f"CNN staged candidate counts differ: {stage_counts}")
    if set(cnn_tuning["run_id"]) != {cnn_selection["baseline"]["run_id"]}:
        raise ValueError("CNN tuning table and selection mix run IDs")
    successful_cnn = cnn_tuning.loc[cnn_tuning["status"].eq("complete")]
    if successful_cnn.empty:
        raise ValueError("CNN has no completed Validation candidates")
    selected_cnn = select_best_candidate(successful_cnn.to_dict("records"))
    if selected_cnn["trial_id"] != cnn_selection["optimized"]["trial"]["trial_id"]:
        raise ValueError(
            "CNN optimized selection is not the Validation-best completed trial"
        )
    if cnn_selection["baseline"]["trial"]["trial_id"] != "cnn_baseline":
        raise ValueError("CNN baseline selection differs from the fixed baseline")
    cnn_best_config = read_json(out / "cnn_best_config.json")
    if (
        cnn_best_config["run_id"] != cnn_selection["baseline"]["run_id"]
        or cnn_best_config["best"]["trial_id"] != selected_cnn["trial_id"]
    ):
        raise ValueError("CNN final best-config file differs from tuning table")
    if mert_checkpoint["baseline"]["run_id"] != mert_checkpoint["optimized"]["run_id"]:
        raise ValueError("MERT variants mix run IDs")
    mert_run_id = mert_checkpoint["baseline"]["run_id"]
    mert_tuning_path = out / f"mert_tuning_{mert_run_id}.csv"
    mert_config_path = out / f"mert_best_config_{mert_run_id}.json"
    mert_threshold_path = out / f"mert_validation_thresholds_{mert_run_id}.json"
    for path in (mert_tuning_path, mert_config_path, mert_threshold_path):
        if not path.is_file():
            raise FileNotFoundError(f"MERT Validation selection is incomplete: {path}")
    mert_tuning = pd.read_csv(mert_tuning_path).sort_values("candidate_order")
    expected_mert = [
        (layer, c) for layer in range(13) for c in (0.01, 0.1, 1.0, 10.0, 100.0)
    ]
    if (
        len(mert_tuning) != len(expected_mert)
        or mert_tuning["candidate_order"].tolist() != list(range(len(expected_mert)))
        or set(mert_tuning["run_id"]) != {mert_run_id}
    ):
        raise ValueError("MERT 13-layer by 5-C Validation search is incomplete")
    for row, (layer, c) in zip(
        mert_tuning.to_dict("records"), expected_mert, strict=True
    ):
        if int(row["layer"]) != layer or not np.isclose(float(row["C"]), c):
            raise ValueError("MERT candidate differs from predeclared layer/C grid")
    mert_best = select_best_candidate(mert_tuning)
    mert_config = read_json(mert_config_path)
    mert_thresholds = read_json(mert_threshold_path)
    if (
        mert_config["run_id"] != mert_run_id
        or mert_thresholds["run_id"] != mert_run_id
        or mert_config["manifest_sha256"] != manifest_hash
        or int(mert_config["candidate_count"]) != len(expected_mert)
    ):
        raise ValueError("MERT saved Validation configuration provenance differs")
    for variant, chosen in (
        ("baseline", (12, 1.0)),
        ("optimized", (int(mert_best["layer"]), float(mert_best["C"]))),
    ):
        payload = mert_checkpoint[variant]
        saved = mert_config[variant]
        if (int(payload["layer"]), float(payload["C"])) != chosen or (
            int(saved["layer"]),
            float(saved["C"]),
        ) != chosen:
            raise ValueError(
                f"MERT {variant} checkpoint is not the frozen Validation choice"
            )
        for level in ("segment", "track"):
            if not np.isclose(
                float(payload["thresholds"][level]),
                float(mert_thresholds[variant][level]),
            ):
                raise ValueError(f"MERT {variant} Validation threshold differs")
    for model in ("LogisticRegression", "RBF-SVM"):
        for variant in ("baseline", "optimized"):
            if not _finite_thresholds(classical_thresholds["models"][model][variant]):
                raise ValueError(f"{model} {variant} Validation thresholds missing")
            config = (classical_baselines if variant == "baseline" else classical_best)[
                "models"
            ][model]
            checkpoint_path = Path(config["checkpoint"])
            stem = "logistic" if model == "LogisticRegression" else "svm"
            kind = "baseline" if variant == "baseline" else "best"
            expected_path = checkpoints / f"{stem}_{kind}.joblib"
            if checkpoint_path.resolve() != expected_path.resolve():
                raise ValueError(f"{model} {variant} checkpoint path mismatch")
            if (
                not checkpoint_path.is_file()
                or file_hash(checkpoint_path) != config["checkpoint_sha256"]
            ):
                raise ValueError(f"{model} {variant} checkpoint hash mismatch")
    if not bool(np.load(logmel_done_path).all()):
        raise ValueError("Log-Mel cache incomplete")
    run_ids = {
        "classical": classical_meta["run_id"],
        "cnn": cnn_selection["baseline"]["run_id"],
        "mert": mert_checkpoint["baseline"]["run_id"],
    }
    return {
        "root": root,
        "out": out,
        "checkpoints": checkpoints,
        "manifest_path": manifest_path,
        "feature_path": feature_path,
        "logmel_path": logmel_path,
        "logmel_index_path": logmel_index_path,
        "manifest_hash": manifest_hash,
        "logmel_index_hash": index_hash,
        "classical_thresholds": classical_thresholds,
        "classical_baselines": classical_baselines,
        "classical_best": classical_best,
        "classical_meta": classical_meta,
        "cnn_selection": cnn_selection,
        "cnn_checkpoint": cnn_checkpoint,
        "mert_checkpoint": mert_checkpoint,
        "run_ids": run_ids,
        "final_run_id": "final_binary_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    }


def load_fixed_test(context: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Open Test only after preflight and align 266-D features by segment ID."""
    manifest = pd.read_csv(context["manifest_path"])
    assert_group_disjoint(
        manifest.rename(columns={"original_audio": "original_audio_id"})
    )
    if manifest["segment_id"].isna().any() or manifest["segment_id"].duplicated().any():
        raise ValueError("Manifest segment ID missing/duplicated")
    valid_mapping = (manifest["label"].eq("REAL") & manifest["label_id"].eq(0)) | (
        manifest["label"].eq("FAKE") & manifest["label_id"].eq(1)
    )
    if not valid_mapping.all():
        raise ValueError("REAL=0/FAKE=1 mapping differs")
    if (
        manifest.groupby("track_sample_id")[["label_id", "original_audio", "split"]]
        .nunique()
        .gt(1)
        .any()
        .any()
    ):
        raise ValueError("Track label/group/split conflict")
    # 사전 고정된 Test 행만 열고 특징은 segment_id로 다시 정렬한다.
    test_meta = manifest.loc[manifest["split"].eq("test")].reset_index(drop=True)
    if test_meta.empty or test_meta["label_id"].nunique() != 2:
        raise ValueError("Test needs both classes")
    feature_columns = context["classical_meta"]["feature_columns"]
    if len(feature_columns) != 266 or len(set(feature_columns)) != 266:
        raise ValueError("Saved feature schema is not 266-D")
    features = pd.read_csv(context["feature_path"])
    if set(features["segment_id"]) != set(manifest["segment_id"]):
        raise ValueError("Feature/manifest segment ID set differs")
    test_features = features.loc[features["split"].eq("test")].copy()
    if set(test_features["segment_id"]) != set(test_meta["segment_id"]):
        raise ValueError("Test feature segment set differs")
    test_features = test_meta[["segment_id"]].merge(
        test_features, on="segment_id", validate="one_to_one"
    )
    matrix = test_features[feature_columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(matrix.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite Test feature")
    return test_meta, matrix, manifest


def score_frame(metadata: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """Attach fixed Test IDs to one model's ordered FAKE scores."""
    frame = (
        metadata[["segment_id", "track_sample_id", "original_audio", "label_id"]]
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
    if len(frame) != len(scores) or not np.isfinite(frame["score"]).all():
        raise ValueError("Score length or finite check failed")
    return frame


def infer_classical(
    context: dict, test_meta: pd.DataFrame, matrix: pd.DataFrame
) -> tuple[dict, list[dict]]:
    """Use saved Train-only scaler pipelines to score LR and SVM."""
    output, times = {}, []
    feature_columns = context["classical_meta"]["feature_columns"]
    for model, stem in (("LogisticRegression", "logistic"), ("RBF-SVM", "svm")):
        reused = {}
        for variant, kind in (("baseline", "baseline"), ("optimized", "best")):
            configs = (
                context["classical_baselines"]
                if variant == "baseline"
                else context["classical_best"]
            )
            candidate = configs["models"][model]["candidate_order"]
            if candidate in reused:
                output[(model, variant)] = reused[candidate]
                times.append(
                    {
                        "model": model,
                        "variant": variant,
                        "seconds": 0.0,
                        "reused_from": "baseline",
                    }
                )
                continue
            # Train에서 학습한 전체 Pipeline을 복원해 Test scaler 재적합을 막는다.
            pipeline = joblib.load(context["checkpoints"] / f"{stem}_{kind}.joblib")
            if list(pipeline.feature_names_in_) != feature_columns:
                raise ValueError(f"{model} feature order differs")
            classifier = pipeline.named_steps["classifier"]
            if not np.array_equal(classifier.classes_, [0, 1]):
                raise ValueError(f"{model} class order differs")
            expected = configs["models"][model]["params"]
            if classifier.C != expected["C"] or classifier.class_weight != "balanced":
                raise ValueError(f"{model} checkpoint parameters differ")
            if model == "RBF-SVM":
                if classifier.gamma != expected["gamma"] or classifier.probability:
                    raise ValueError("SVM gamma/probability setting differs")
            started = time.perf_counter()
            if model == "LogisticRegression":
                fake_column = int(np.flatnonzero(classifier.classes_ == 1)[0])
                scores = pipeline.predict_proba(matrix)[:, fake_column]
            else:
                scores = pipeline.decision_function(matrix)
            frame = score_frame(test_meta, scores)
            output[(model, variant)] = frame
            reused[candidate] = frame
            times.append(
                {
                    "model": model,
                    "variant": variant,
                    "seconds": time.perf_counter() - started,
                    "reused_from": None,
                }
            )
    return output, times


def infer_cnn(
    context: dict, test_meta: pd.DataFrame, manifest: pd.DataFrame
) -> tuple[dict, list[dict], str]:
    """Score fixed CNN checkpoints on the verified original Log-Mel cache."""
    cache_index = pd.read_csv(context["logmel_index_path"])
    if (
        not cache_index["segment_id"]
        .reset_index(drop=True)
        .equals(manifest["segment_id"].reset_index(drop=True))
    ):
        raise ValueError("Log-Mel cache index order differs from manifest")
    cache = np.load(context["logmel_path"], mmap_mode="r")
    if cache.shape != (len(manifest), 128, 1001):
        raise ValueError(f"Unexpected Log-Mel cache shape {cache.shape}")
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else (
            "mps"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
    )
    output, times, reused = {}, [], {}
    for variant in ("baseline", "optimized"):
        trial_id = context["cnn_selection"][variant]["trial"]["trial_id"]
        if trial_id in reused:
            output[("Log-Mel CNN", variant)] = reused[trial_id]
            times.append(
                {
                    "model": "Log-Mel CNN",
                    "variant": variant,
                    "seconds": 0.0,
                    "reused_from": "baseline",
                }
            )
            continue
        checkpoint = context["cnn_checkpoint"][variant]
        config = checkpoint["config"]
        model = LogMelCNN(dropout=float(config["dropout"])).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        dataset = LogMelDataset(test_meta, context["logmel_path"], cache_index)
        loader = DataLoader(
            dataset,
            batch_size=int(config["batch_size"]),
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        started = time.perf_counter()
        frame = predict_scores(model, loader, test_meta, device)
        output[("Log-Mel CNN", variant)] = frame
        reused[trial_id] = frame
        times.append(
            {
                "model": "Log-Mel CNN",
                "variant": variant,
                "seconds": time.perf_counter() - started,
                "reused_from": None,
            }
        )
    return output, times, str(device)


def infer_mert(context: dict, test_meta: pd.DataFrame) -> tuple[dict, list[dict], dict]:
    """Extract Test embeddings once after tuning, then score frozen LR heads."""
    # MERT Baseline·Optimized도 저장된 layer와 LR head만 사용한다.
    for variant, payload in context["mert_checkpoint"].items():
        cache_meta = payload["cache_meta"]
        for key, expected in (
            ("revision", MODEL_REVISION),
            ("processor_sampling_rate", 24000),
            ("pooling", "single-segment valid hidden frames mean"),
            ("manifest_sha256", context["manifest_hash"]),
        ):
            if cache_meta[key] != expected:
                raise ValueError(f"MERT {variant} cache provenance differs at {key}")
    # 이 환경에서는 sklearn 선행 import 후 custom MERT load가 native segfault를
    # 일으킨다. Test embedding 추출은 sklearn이 없는 별도 프로세스에서 수행한다.
    subprocess.run(
        [
            sys.executable,
            str(context["root"] / "scripts/extract_mert_test.py"),
            "--root",
            str(context["root"]),
            "--manifest-sha256",
            context["manifest_hash"],
        ],
        check=True,
        cwd=context["root"],
    )
    cache_dir = context["root"] / "data/processed/mert"
    stem = f"mert95m_binary_test_validframe_{MODEL_REVISION[:12]}"
    embedding_path = cache_dir / f"{stem}.npy"
    done_path = cache_dir / f"{stem}_done.npy"
    index_path = cache_dir / f"{stem}_index.csv"
    metadata_path = cache_dir / f"{stem}_meta.json"
    summary_path = cache_dir / f"{stem}_meta_run.json"
    summary = read_json(summary_path)
    expected_meta = embedding_cache_metadata(test_meta, context["manifest_path"])
    if (
        read_json(metadata_path) != expected_meta
        or summary["metadata"] != expected_meta
    ):
        raise ValueError("MERT Test cache provenance mismatch")
    index = pd.read_csv(index_path)
    if (
        index["segment_id"].astype(str).tolist()
        != test_meta["segment_id"].astype(str).tolist()
    ):
        raise ValueError("MERT Test cache index differs from fixed Test")
    if not bool(np.load(done_path).all()):
        raise ValueError("MERT Test cache incomplete")
    embeddings = np.load(embedding_path, mmap_mode="r")
    if embeddings.shape != (len(test_meta), 13, 768):
        raise ValueError("MERT Test embeddings shape mismatch")
    output, times, reused = {}, [], {}
    for variant in ("baseline", "optimized"):
        payload = context["mert_checkpoint"][variant]
        candidate = (int(payload["layer"]), float(payload["C"]))
        if candidate in reused:
            output[("Frozen MERT + LR", variant)] = reused[candidate]
            times.append(
                {
                    "model": "Frozen MERT + LR",
                    "variant": variant,
                    "seconds": 0.0,
                    "reused_from": "baseline",
                }
            )
            continue
        values = np.asarray(embeddings[:, candidate[0], :], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("MERT Test embedding contains nonfinite values")
        classifier = payload["classifier"]
        if not np.array_equal(classifier.classes_, [0, 1]):
            raise ValueError("MERT LR class order differs")
        started = time.perf_counter()
        scaled = payload["scaler"].transform(values)
        fake_column = int(np.flatnonzero(classifier.classes_ == 1)[0])
        frame = score_frame(test_meta, classifier.predict_proba(scaled)[:, fake_column])
        output[("Frozen MERT + LR", variant)] = frame
        reused[candidate] = frame
        times.append(
            {
                "model": "Frozen MERT + LR",
                "variant": variant,
                "seconds": time.perf_counter() - started,
                "reused_from": None,
            }
        )
    details = {
        "device": summary["device"],
        "model_load_seconds": summary["model_load_seconds"],
        "embedding_extraction_seconds": summary["embedding_extraction_seconds"],
        "cache_paths": summary["paths"],
        "cache_metadata": expected_meta,
    }
    return output, times, details


# 입력: 고정 Test 정보·여덟 모델 점수·추론 시간.
# 출력: 전체 지표, Track/Segment 비교표와 네 Optimized 모델 표.
def finish_test(
    context: dict,
    test_meta: pd.DataFrame,
    scores: dict,
    inference_times: list[dict],
    cnn_device: str,
    mert_details: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate all eight variants, save Test tables/raw scores and plots."""
    thresholds = {}
    for model in ("LogisticRegression", "RBF-SVM"):
        for variant in ("baseline", "optimized"):
            thresholds[(model, variant)] = context["classical_thresholds"]["models"][
                model
            ][variant]
    for variant in ("baseline", "optimized"):
        thresholds[("Log-Mel CNN", variant)] = context["cnn_selection"][variant][
            "thresholds"
        ]
        thresholds[("Frozen MERT + LR", variant)] = context["mert_checkpoint"][variant][
            "thresholds"
        ]
    if set(scores) != set(thresholds) or len(scores) != 8:
        raise ValueError("Exactly eight frozen model variants are required")
    model_order = ("LogisticRegression", "RBF-SVM", "Log-Mel CNN", "Frozen MERT + LR")
    metric_rows = []
    for model in model_order:
        for variant in ("baseline", "optimized"):
            source = (
                context["run_ids"]["classical"]
                if model in model_order[:2]
                else (
                    context["run_ids"]["cnn"]
                    if model == "Log-Mel CNN"
                    else context["run_ids"]["mert"]
                )
            )
            metric_rows.extend(
                evaluate_frozen_test_run(
                    test_meta,
                    scores[(model, variant)],
                    thresholds[(model, variant)],
                    model=model,
                    variant=variant,
                    run_id=source,
                    output_dir=context["out"] / "test_scores",
                )
            )
    metrics = pd.DataFrame(metric_rows)
    track, segment, optimized = comparison_tables(metrics)
    best_hyperparameters = {
        "LogisticRegression": {
            "C": context["classical_best"]["models"]["LogisticRegression"]["params"][
                "C"
            ]
        },
        "RBF-SVM": {
            key: context["classical_best"]["models"]["RBF-SVM"]["params"][key]
            for key in ("C", "gamma")
        },
        "Log-Mel CNN": {
            key: context["cnn_selection"]["optimized"]["trial"][key]
            for key in (
                "lr",
                "dropout",
                "weight_decay",
                "batch_size",
                "max_epochs",
                "patience",
            )
        },
        "Frozen MERT + LR": {
            key: context["mert_checkpoint"]["optimized"][key] for key in ("layer", "C")
        },
    }
    for table in (track, segment, optimized):
        table["best_hyperparameters"] = table["model"].map(
            lambda model: json.dumps(
                best_hyperparameters[model], ensure_ascii=False, default=str
            )
        )
    for table in (metrics, track, segment, optimized):
        table.insert(0, "final_run_id", context["final_run_id"])
    out = context["out"]
    # 최종 수치와 원점수를 별도로 저장해 보고서 값을 추적할 수 있게 한다.
    metrics.to_csv(
        out / "optimized_test_results.csv", index=False, encoding="utf-8-sig"
    )
    track.to_csv(out / "baseline_vs_optimized.csv", index=False, encoding="utf-8-sig")
    segment.to_csv(
        out / "baseline_vs_optimized_segment.csv", index=False, encoding="utf-8-sig"
    )
    optimized.to_csv(
        out / "optimized_four_model_comparison.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(inference_times).to_csv(out / "final_inference_times.csv", index=False)
    for name, content in (
        (
            "validation_thresholds.json",
            {
                f"{model}|{variant}": value
                for (model, variant), value in thresholds.items()
            },
        ),
        (
            "baseline_configs.json",
            {
                "classical": context["classical_baselines"]["models"],
                "cnn": context["cnn_selection"]["baseline"]["trial"],
                "mert": {
                    key: context["mert_checkpoint"]["baseline"][key]
                    for key in ("layer", "C")
                },
            },
        ),
        (
            "best_configs.json",
            {
                "classical": context["classical_best"]["models"],
                "cnn": context["cnn_selection"]["optimized"]["trial"],
                "mert": {
                    key: context["mert_checkpoint"]["optimized"][key]
                    for key in ("layer", "C")
                },
            },
        ),
    ):
        (out / name).write_text(
            json.dumps(
                {
                    "final_run_id": context["final_run_id"],
                    "component_runs": context["run_ids"],
                    "split_manifest_sha256": context["manifest_hash"],
                    "models": content,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
    (out / "final_run_metadata.json").write_text(
        json.dumps(
            {
                "final_run_id": context["final_run_id"],
                "component_runs": context["run_ids"],
                "split_manifest_sha256": context["manifest_hash"],
                "test_segments": len(test_meta),
                "mert": mert_details,
                "cnn_device": cnn_device,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "sklearn": sklearn.__version__,
                "torch": torch.__version__,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    order = list(model_order)
    plotted = track.set_index("model").loc[order]
    for metric, title, filename in (
        ("roc_auc", "Track Test ROC-AUC", "baseline_vs_optimized_auc.png"),
        ("eer", "Track Test EER", "baseline_vs_optimized_eer.png"),
    ):
        x = np.arange(len(order))
        fig, axis = plt.subplots(figsize=(10, 5))
        axis.bar(x - 0.18, plotted[f"baseline_{metric}"], 0.36, label="Baseline")
        axis.bar(x + 0.18, plotted[f"optimized_{metric}"], 0.36, label="Optimized")
        axis.set_xticks(x, order, rotation=15, ha="right")
        axis.set_ylabel(metric)
        axis.set_title(title)
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / filename, dpi=160)
        plt.close(fig)
    return metrics, track, segment, optimized
