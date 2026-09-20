"""Fresh filtered-Train fits with main-run hyperparameters fixed in advance.

This primary follow-up is a fixed-hyperparameter transfer experiment. The
original main-run configuration search included both target generators, and
the original Test has been viewed historically. This script never retunes C,
gamma or MERT layer on the filtered Validation set.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.modeling_evaluation import evaluate_binary_predictions
from scripts.run_unseen_classical_mert import (
    ROOT,
    MERT_REVISION,
    MERT_TRAIN_STEM,
    MERT_TEST_STEM,
    SEED,
    TARGETS,
    file_sha256,
    filtered_split,
    log,
    read_manifest,
    read_mert_cache,
    read_test_features,
    read_training_features,
    score_frame,
    score_frozen_model,
    test_subset,
    paired_subset,
)

RESULT_ROOT = ROOT / "results/unseen_fixed_transfer_classical_mert"
CHECKPOINT_ROOT = ROOT / "checkpoints/unseen_fixed_transfer_classical_mert"
PROTOCOL_PATH = ROOT / "docs/UNSEEN_FIXED_PROTOCOL.md"
PROTOCOL_SHA256 = "098fa65192f65fddbf7d65757eb5c6b5c2b1149f79178c27746b4e328ee54a5e"
MODELS = ("LogisticRegression", "RBF-SVM", "Frozen MERT + LR")
VARIANT = "fixed_hyperparameter_transfer"


# Main의 고정 하이퍼파라미터와 checkpoint 출처를 확인한다.
def check_main_configs(manifest_hash: str) -> dict:
    """Verify fixed settings against the saved main-run selections."""
    classical_path = ROOT / "results/model_tuning/classical_best_configs.json"
    mert_path = (
        ROOT / "results/model_tuning/mert_best_config_mert_binary_20260919T110741Z.json"
    )
    classical = json.loads(classical_path.read_text(encoding="utf-8"))
    mert = json.loads(mert_path.read_text(encoding="utf-8"))
    if (
        classical["split_manifest_sha256"] != manifest_hash
        or mert["manifest_sha256"] != manifest_hash
        or mert["revision"] != MERT_REVISION
    ):
        raise ValueError(
            "Main-run selection source differs from original manifest/MERT"
        )
    lr = classical["models"]["LogisticRegression"]
    svm = classical["models"]["RBF-SVM"]
    if (
        lr["params"]["C"] != 0.01
        or svm["params"]["C"] != 10.0
        or svm["params"]["gamma"] != 0.001
        or mert["optimized"]["layer"] != 4
        or mert["optimized"]["C"] != 0.01
    ):
        raise ValueError("Main-run best settings differ from fixed protocol")
    for entry in (lr, svm):
        if file_sha256(Path(entry["checkpoint"])) != entry["checkpoint_sha256"]:
            raise ValueError("Main classical selection checkpoint hash differs")
    mert_checkpoint = Path(mert["optimized"]["checkpoint"])
    mert_payload = joblib.load(mert_checkpoint)
    if (
        mert_payload["run_id"] != mert["run_id"]
        or mert_payload["revision"] != MERT_REVISION
        or mert_payload["layer"] != 4
        or mert_payload["C"] != 0.01
    ):
        raise ValueError("Main MERT optimized checkpoint differs from fixed selection")
    return {
        "source_runs": {"classical": classical["run_id"], "mert": mert["run_id"]},
        "source_config_sha256": {
            "classical": file_sha256(classical_path),
            "mert": file_sha256(mert_path),
        },
        "source_checkpoint_sha256": {
            "LogisticRegression": lr["checkpoint_sha256"],
            "RBF-SVM": svm["checkpoint_sha256"],
            "Frozen MERT + LR": file_sha256(mert_checkpoint),
        },
        "main_best": {
            "LogisticRegression": {"C": 0.01},
            "RBF-SVM": {"C": 10.0, "gamma": 0.001},
            "Frozen MERT + LR": {"layer": 4, "C": 0.01},
        },
    }


# 한 holdout·모델 조합을 새로 학습하고 임계값만 필터된 Validation에서 고른다.
# 입력: 한 생성기 제외 Train/Validation과 Main에서 고정한 모델 설정.
# 출력: 새 checkpoint, Validation 점수·임계값의 선택 정보.
def fit_one(
    target: str,
    model_name: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: pd.DataFrame,
    columns: list[str],
    train_val: pd.DataFrame,
    embeddings: np.ndarray,
    checkpoint: Path,
    run_id: str,
    output: Path,
) -> dict:
    """Fit exactly one fixed candidate and choose only Validation thresholds."""
    y_train = train["label_id"].to_numpy(dtype=int)
    if model_name in MODELS[:2]:
        # 필터된 Train/Validation ID에 feature를 1:1로 붙여 행 순서 혼입을 막는다.
        x_train = train[["segment_id"]].merge(
            features, on="segment_id", validate="one_to_one"
        )[columns]
        x_val = val[["segment_id"]].merge(
            features, on="segment_id", validate="one_to_one"
        )[columns]
        if len(x_train) != len(train) or len(x_val) != len(val):
            raise ValueError("Filtered classical feature alignment differs")
        classifier = (
            LogisticRegression(
                C=0.01,
                solver="lbfgs",
                penalty="l2",
                class_weight="balanced",
                max_iter=3000,
                random_state=SEED,
            )
            if model_name == "LogisticRegression"
            else SVC(
                kernel="rbf",
                C=10.0,
                gamma=0.001,
                class_weight="balanced",
                probability=False,
                cache_size=2048,
                random_state=SEED,
            )
        )
        model = Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])
        started = time.perf_counter()
        # scaler도 Pipeline 안에서 Train에만 fit된다.
        model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - started
        if not np.array_equal(classifier.classes_, [0, 1]):
            raise ValueError("Class order differs from REAL=0/FAKE=1")
        scores = (
            model.predict_proba(x_val)[:, 1]
            if model_name == "LogisticRegression"
            else model.decision_function(x_val)
        )
        joblib.dump(model, checkpoint)
    else:
        # MERT encoder는 동결된 표현이며 layer 4는 Main 실험에서 이미 선택됐다.
        positions = pd.Index(train_val["segment_id"])
        train_idx = positions.get_indexer(train["segment_id"])
        val_idx = positions.get_indexer(val["segment_id"])
        if (train_idx < 0).any() or (val_idx < 0).any():
            raise ValueError("Filtered MERT cache alignment differs")
        x_train = np.asarray(embeddings[train_idx, 4, :], dtype=np.float32)
        x_val = np.asarray(embeddings[val_idx, 4, :], dtype=np.float32)
        if not np.isfinite(x_train).all() or not np.isfinite(x_val).all():
            raise ValueError("Filtered MERT embeddings contain invalid values")
        started = time.perf_counter()
        # MERT의 768차원 scaler와 LR도 대상 제외 Train으로 새로 학습한다.
        scaler = StandardScaler().fit(x_train)
        classifier = LogisticRegression(
            C=0.01,
            solver="lbfgs",
            penalty="l2",
            class_weight="balanced",
            max_iter=3000,
            random_state=SEED,
        )
        classifier.fit(scaler.transform(x_train), y_train)
        fit_seconds = time.perf_counter() - started
        if not np.array_equal(classifier.classes_, [0, 1]):
            raise ValueError("MERT LR class order differs")
        scores = classifier.predict_proba(scaler.transform(x_val))[:, 1]
        joblib.dump(
            {
                "run_id": run_id,
                "holdout_generator": target,
                "model": "m-a-p/MERT-v1-95M",
                "revision": MERT_REVISION,
                "layer": 4,
                "C": 0.01,
                "scaler": scaler,
                "classifier": classifier,
                "feature_dim": 768,
            },
            checkpoint,
        )
    frame = score_frame(val, scores)
    # 이 호출이 필터된 Validation의 Segment/Track 임계값을 각각 결정한다.
    report = evaluate_binary_predictions(frame)
    for level in ("segment", "track"):
        raw = report[f"{level}_scores"].copy()
        raw.insert(0, "threshold", report["thresholds"][level])
        raw.insert(0, "level", level)
        raw.insert(0, "split", "val")
        raw.insert(0, "model", model_name)
        raw.insert(0, "holdout_generator", target)
        raw.insert(0, "run_id", run_id)
        safe = (
            model_name.lower().replace(" ", "_").replace("+", "plus").replace("-", "_")
        )
        raw.to_csv(
            output / f"{target}_{safe}_{level}_val_scores.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return {
        "model": model_name,
        "train_segments": len(train),
        "val_segments": len(val),
        "fit_seconds": fit_seconds,
        "thresholds": report["thresholds"],
        "val_segment": {
            k: v for k, v in report["segment"].items() if k != "confusion_matrix"
        },
        "val_track": {
            k: v for k, v in report["track"].items() if k != "confusion_matrix"
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
    }


def metric_rows(
    report: dict, run_id: str, target: str, model: str, cohort: str
) -> list[dict]:
    return [
        {
            "run_id": run_id,
            "holdout_generator": target,
            "model": model,
            "variant": VARIANT,
            "cohort": cohort,
            "split": "test",
            "level": level,
            "validation_segment_threshold": report["thresholds"]["segment"],
            "validation_track_threshold": report["thresholds"]["track"],
            **report[level],
        }
        for level in ("segment", "track")
    ]


def main() -> None:
    started = time.perf_counter()
    # 고정 설정 후속 프로토콜이 바뀌면 학습·Test 평가를 중지한다.
    if file_sha256(PROTOCOL_PATH) != PROTOCOL_SHA256:
        raise ValueError("New primary fixed-transfer protocol SHA-256 changed")
    manifest, manifest_hash = read_manifest()
    main_config = check_main_configs(manifest_hash)
    run_id = "unseen_fixed_classical_mert_" + datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    output = RESULT_ROOT / run_id
    checkpoint_dir = CHECKPOINT_ROOT / run_id
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    columns = json.loads(
        (ROOT / "results/model_tuning/classical_run_metadata.json").read_text()
    )["feature_columns"]
    features = read_training_features(manifest, columns)
    train_val = manifest.loc[manifest["split"].isin(("train", "val"))].reset_index(
        drop=True
    )
    embeddings, train_cache_meta = read_mert_cache(
        MERT_TRAIN_STEM, train_val, manifest_hash
    )
    selections: dict[str, dict] = {}
    inclusion = []
    for target in TARGETS:
        train = filtered_split(train_val, target, "train")
        val = filtered_split(train_val, target, "val")
        selections[target] = {}
        for split_name, part in (("train", train), ("val", val)):
            for label_id, label in ((0, "REAL"), (1, "FAKE")):
                rows = part.loc[part["label_id"].eq(label_id)]
                inclusion.append(
                    {
                        "run_id": run_id,
                        "holdout_generator": target,
                        "split": split_name,
                        "label": label,
                        "original_audio": rows["original_audio"].nunique(),
                        "tracks": rows["track_sample_id"].nunique(),
                        "segments": len(rows),
                    }
                )
        for model, stem in (
            ("LogisticRegression", "lr"),
            ("RBF-SVM", "svm"),
            ("Frozen MERT + LR", "mert"),
        ):
            path = checkpoint_dir / f"{target}_{stem}_fixed.joblib"
            choice = fit_one(
                target,
                model,
                train,
                val,
                features,
                columns,
                train_val,
                embeddings,
                path,
                run_id,
                output,
            )
            selections[target][model] = choice
            log(
                f"fixed {target} {model}: Val Track EER={choice['val_track']['eer']:.6f} "
                f"AUC={choice['val_track']['roc_auc']:.6f}"
            )
    pd.DataFrame(inclusion).to_csv(
        output / "inclusion_counts.csv", index=False, encoding="utf-8-sig"
    )
    choice_path = output / "fixed_selections.json"
    choice_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "variant": VARIANT,
                "protocol_sha256": PROTOCOL_SHA256,
                "manifest_sha256": manifest_hash,
                **main_config,
                "selected": selections,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    # 여섯 새 모델과 각각의 Validation 임계값을 기록한 뒤 Test를 처음 연다.
    choice_hash = file_sha256(choice_path)
    log("Six fresh fits and Validation thresholds frozen; opening original Test")

    test_all = manifest.loc[manifest["split"].eq("test")].reset_index(drop=True)
    test_embeddings, test_cache_meta = read_mert_cache(
        MERT_TEST_STEM, test_all, manifest_hash
    )
    test_index = pd.Index(test_all["segment_id"])
    metrics, paired_metrics, raw_rows, cohorts = [], [], [], []
    for target in TARGETS:
        primary = test_subset(manifest, target)
        paired = paired_subset(primary)
        fake_groups = primary.loc[primary.label_id.eq(1), "original_audio"].nunique()
        cohorts.append(
            {
                "run_id": run_id,
                "holdout_generator": target,
                "primary_real_tracks": primary.loc[
                    primary.label_id.eq(0), "track_sample_id"
                ].nunique(),
                "primary_fake_tracks": primary.loc[
                    primary.label_id.eq(1), "track_sample_id"
                ].nunique(),
                "primary_fake_original_audio": fake_groups,
                "primary_segments": len(primary),
                "paired_real_tracks": paired.loc[
                    paired.label_id.eq(0), "track_sample_id"
                ].nunique(),
                "paired_fake_tracks": paired.loc[
                    paired.label_id.eq(1), "track_sample_id"
                ].nunique(),
            }
        )
        x_test = read_test_features(primary, columns)
        positions = test_index.get_indexer(primary["segment_id"])
        if (positions < 0).any():
            raise ValueError("MERT Test cache lacks primary segment")
        mert_test = test_embeddings[positions]
        for model in MODELS:
            choice = selections[target][model]
            checkpoint = Path(choice["checkpoint"])
            if file_sha256(checkpoint) != choice["checkpoint_sha256"]:
                raise ValueError("Frozen fixed-transfer checkpoint hash changed")
            frame, _ = score_frozen_model(model, checkpoint, primary, x_test, mert_test)
            # Test 점수로 재튜닝하지 않고 저장된 임계값을 그대로 적용한다.
            report = evaluate_binary_predictions(frame, thresholds=choice["thresholds"])
            metrics.extend(metric_rows(report, run_id, target, model, "primary"))
            for level in ("segment", "track"):
                raw = report[f"{level}_scores"].copy()
                for key, value in (
                    ("run_id", run_id),
                    ("holdout_generator", target),
                    ("model", model),
                    ("variant", VARIANT),
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
                metric_rows(paired_report, run_id, target, model, "paired_source")
            )
            log(
                f"fixed Test {target} {model}: Track AUC={report['track']['roc_auc']:.6f} "
                f"EER={report['track']['eer']:.6f}"
            )
    # primary와 paired-source 결과를 나눠 저장해 평가 모집단을 구분한다.
    pd.DataFrame(metrics).to_csv(
        output / "primary_test_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(paired_metrics).to_csv(
        output / "paired_source_test_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(raw_rows, ignore_index=True).to_csv(
        output / "primary_test_scores.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(cohorts).to_csv(
        output / "test_cohorts.csv", index=False, encoding="utf-8-sig"
    )
    (output / "run_metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "variant": VARIANT,
                "protocol_sha256": PROTOCOL_SHA256,
                "manifest_sha256": manifest_hash,
                "fixed_selections_sha256": choice_hash,
                "source_config_sha256": main_config["source_config_sha256"],
                "source_checkpoint_sha256": main_config["source_checkpoint_sha256"],
                "source_runs": main_config["source_runs"],
                "checkpoint_sha256": {
                    t: {m: selections[t][m]["checkpoint_sha256"] for m in MODELS}
                    for t in TARGETS
                },
                "test_cohorts": cohorts,
                "mert_train_val_cache": train_cache_meta,
                "mert_test_cache": test_cache_meta,
                "feature_columns_sha256": hashlib.sha256(
                    "\n".join(columns).encode()
                ).hexdigest(),
                "hyperparameters_reselected_on_filtered_val": False,
                "thresholds_selected_on_filtered_val": True,
                "historical_exposure": (
                    "Main best hyperparameters were selected with MusicGen/Udio in Validation; "
                    "the original Test and target generators were examined in prior work. "
                    "This is fixed-hyperparameter transfer, not fully unseen selection."
                ),
                "versions": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                    "scikit_learn": sklearn.__version__,
                },
                "elapsed_seconds": time.perf_counter() - started,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    (RESULT_ROOT / "latest_run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "result_dir": str(output),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"Fixed-transfer run complete: {run_id}, {time.perf_counter()-started:.1f}s")


if __name__ == "__main__":
    main()
