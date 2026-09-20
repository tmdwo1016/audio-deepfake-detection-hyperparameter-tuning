"""Final-stage Test safeguards and comparison tables for revised binary runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.modeling_evaluation import evaluate_binary_predictions, validate_segment_frame


def verify_test_score_frame(
    test_manifest: pd.DataFrame, scored: pd.DataFrame
) -> pd.DataFrame:
    """Require exactly the fixed Test segments and their original metadata."""
    required = {"segment_id", "track_sample_id", "original_audio", "label_id", "split"}
    if not required.issubset(test_manifest):
        raise ValueError(
            f"Test manifest missing {sorted(required - set(test_manifest))}"
        )
    if not test_manifest["split"].eq("test").all():
        raise ValueError("Final evaluation received non-Test manifest rows")
    # 예측 표를 고정 Test manifest의 ID·원곡·라벨과 일대일 대조한다.
    expected = (
        test_manifest[["segment_id", "track_sample_id", "original_audio", "label_id"]]
        .rename(
            columns={
                "track_sample_id": "track_id",
                "original_audio": "original_audio_id",
                "label_id": "label",
            }
        )
        .copy()
    )
    if expected["segment_id"].isna().any() or expected["segment_id"].duplicated().any():
        raise ValueError("Test manifest has null or duplicate segment_id")
    actual = validate_segment_frame(scored)
    if set(actual["segment_id"]) != set(expected["segment_id"]) or len(actual) != len(
        expected
    ):
        missing = set(expected["segment_id"]) - set(actual["segment_id"])
        extra = set(actual["segment_id"]) - set(expected["segment_id"])
        raise ValueError(
            f"Test segment mismatch: {len(missing)} missing, {len(extra)} extra"
        )
    merged = expected.merge(
        actual,
        on="segment_id",
        how="left",
        validate="one_to_one",
        suffixes=("_expected", ""),
    )
    for column in ("track_id", "original_audio_id", "label"):
        if not merged[column].eq(merged[f"{column}_expected"]).all():
            raise ValueError(f"Test {column} differs from fixed manifest")
    return merged[["segment_id", "track_id", "original_audio_id", "label", "score"]]


def evaluate_frozen_test_run(
    test_manifest: pd.DataFrame,
    scored: pd.DataFrame,
    validation_thresholds: dict[str, float],
    *,
    model: str,
    variant: str,
    run_id: str,
    output_dir: str | Path,
) -> list[dict]:
    """Apply saved Validation thresholds and save both levels of Test scores."""
    if variant not in {"baseline", "optimized"}:
        raise ValueError("variant must be baseline or optimized")
    if not run_id or not model:
        raise ValueError("run_id and model are required")
    if set(validation_thresholds) != {"segment", "track"} or not all(
        np.isfinite(float(value)) for value in validation_thresholds.values()
    ):
        raise ValueError("Both finite saved Validation thresholds are required")
    # Test 점수의 누락·중복을 먼저 검사한 뒤 저장된 Validation 임계값만 적용한다.
    verified = verify_test_score_frame(test_manifest, scored)
    evaluation = evaluate_binary_predictions(verified, thresholds=validation_thresholds)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    safe_model = model.lower().replace(" ", "_").replace("-", "_")
    for level in ("segment", "track"):
        raw = evaluation[f"{level}_scores"].copy()
        raw.insert(0, "run_id", run_id)
        raw.insert(1, "model", model)
        raw.insert(2, "variant", variant)
        raw.insert(3, "split", "test")
        raw.insert(4, "level", level)
        raw["threshold"] = validation_thresholds[level]
        # 재계산 가능한 Segment·Track 원점수와 적용 임계값을 함께 보관한다.
        raw.to_csv(
            destination / f"{safe_model}_{variant}_test_{level}_scores.csv",
            index=False,
            encoding="utf-8-sig",
        )
        metrics = evaluation[level].copy()
        metrics["confusion_matrix"] = json.dumps(metrics["confusion_matrix"])
        rows.append(
            {
                "run_id": run_id,
                "model": model,
                "variant": variant,
                "split": "test",
                "level": level,
                **metrics,
            }
        )
    return rows


def comparison_tables(
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return track, segment baseline comparisons and optimized four-model rows."""
    required = {
        "model",
        "variant",
        "split",
        "level",
        "roc_auc",
        "eer",
        "balanced_accuracy",
        "macro_f1",
        "ap_fake",
        "ap_real",
        "real_fpr",
        "fake_miss_rate",
        "hter",
    }
    if not required.issubset(metrics):
        raise ValueError(f"Metrics missing {sorted(required - set(metrics))}")
    if not metrics["split"].eq("test").all():
        raise ValueError("Comparison tables require Test-only rows")
    if metrics.duplicated(["model", "variant", "level"]).any():
        raise ValueError("Duplicate model/variant/level Test metrics")
    models = set(metrics["model"])
    if len(models) != 4 or len(metrics) != 16:
        raise ValueError("Exactly four models × two variants × two levels are required")
    measures = [
        "roc_auc",
        "eer",
        "balanced_accuracy",
        "macro_f1",
        "ap_fake",
        "ap_real",
        "real_fpr",
        "fake_miss_rate",
        "hter",
    ]
    # Baseline과 Optimized의 차이를 같은 모델·평가 수준에서만 계산한다.
    comparisons = {}
    for level in ("track", "segment"):
        subset = metrics.loc[metrics["level"].eq(level)]
        if set(subset["variant"]) != {"baseline", "optimized"}:
            raise ValueError(f"Both variants missing at {level} level")
        rows = []
        for model in sorted(models):
            pair = subset.set_index(["model", "variant"])
            if (model, "baseline") not in pair.index or (
                model,
                "optimized",
            ) not in pair.index:
                raise ValueError(f"Missing {model} variant at {level} level")
            row = {"model": model, "level": level}
            for measure in measures:
                baseline = float(pair.loc[(model, "baseline"), measure])
                optimized = float(pair.loc[(model, "optimized"), measure])
                row[f"baseline_{measure}"] = baseline
                row[f"optimized_{measure}"] = optimized
                row[f"delta_{measure}"] = optimized - baseline
            rows.append(row)
        comparisons[level] = pd.DataFrame(rows)
    optimized = metrics.loc[metrics["variant"].eq("optimized")].copy()
    return comparisons["track"], comparisons["segment"], optimized
