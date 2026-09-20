"""Verify frozen unseen-generator artifacts and recompute Test metrics.

This is a read-only audit of saved predictions. It never fits a model or
chooses a threshold from Test.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "unseen_classical_mert_20260919T171142Z"
RESULT = ROOT / "results/unseen_revised_classical_mert" / RUN_ID
MODELS = ("LogisticRegression", "RBF-SVM", "Frozen MERT + LR")
TARGETS = ("musicgen", "udio")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def near(actual: float, expected: float, field: str) -> None:
    if not np.isclose(actual, expected, atol=1e-12, rtol=1e-12):
        raise AssertionError(f"{field}: {actual} != {expected}")


# 저장된 원점수에서 분류·순위 지표를 별도로 다시 계산한다.
# 입력: 저장된 원점수 표와 Validation 임계값. 출력: 독립 계산한 지표 사전.
def independent_metrics(frame: pd.DataFrame, threshold: float) -> dict:
    y = frame["label"].to_numpy(dtype=int)
    score = frame["score"].to_numpy(dtype=float)
    prediction = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y, score, drop_intermediate=False)
    delta = fpr - (1 - tpr)
    zero = np.flatnonzero(delta == 0)
    if len(zero):
        eer = float(fpr[zero[0]])
    else:
        crossing = np.flatnonzero(delta[:-1] * delta[1:] < 0)
        if len(crossing) != 1:
            raise AssertionError("EER crossing missing or ambiguous")
        k = crossing[0]
        eer = float(np.interp(0, [delta[k], delta[k + 1]], [fpr[k], fpr[k + 1]]))
    real_fpr = fp / (tn + fp)
    fake_miss = fn / (fn + tp)
    return {
        "n": len(y),
        "n_real": int((y == 0).sum()),
        "n_fake": int((y == 1).sum()),
        "roc_auc": roc_auc_score(y, score),
        "ap_fake": average_precision_score(y, score),
        "ap_real": average_precision_score(1 - y, -score),
        "eer": eer,
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
        "macro_f1": f1_score(y, prediction, average="macro"),
        "real_fpr": real_fpr,
        "fake_miss_rate": fake_miss,
        "hter": (real_fpr + fake_miss) / 2,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def main() -> None:
    manifest = pd.read_csv(ROOT / "data/metadata/segment_manifest_10s.csv")
    tuning = pd.read_csv(RESULT / "tuning.csv")
    selected = json.loads((RESULT / "selected_configs.json").read_text())
    meta = json.loads((RESULT / "run_metadata.json").read_text())
    scores = pd.read_csv(RESULT / "primary_test_scores.csv")
    metrics = pd.read_csv(RESULT / "primary_test_metrics.csv")
    paired = pd.read_csv(RESULT / "paired_source_test_metrics.csv")
    cohort = pd.read_csv(RESULT / "test_cohorts.csv")
    # 실행 당시 프로토콜·manifest·checkpoint 해시를 대조한다.
    protocol_hash = sha256(ROOT / "docs/ROBUSTNESS_PROTOCOL.md")
    assert selected["protocol_sha256"] == meta["protocol_sha256"] == protocol_hash
    assert (
        protocol_hash
        == "95f1546ce8e3fcedb66e5ef52488c9255e414965514b5163e1a62b9052e4941e"
    )
    assert (
        selected["manifest_sha256"]
        == meta["manifest_sha256"]
        == sha256(ROOT / "data/metadata/segment_manifest_10s.csv")
    )
    assert len(tuning) == 172 and len(metrics) == len(paired) == 12
    assert set(scores["run_id"]) == set(tuning["run_id"]) == {RUN_ID}
    assert set(scores["cohort"]) == {"primary"} and set(scores["split"]) == {"test"}
    assert set(scores["variant"]) == {"optimized"}
    assert not tuning["warnings"].ne("[]").any()
    assert meta["selection_used_test"] is False
    assert meta["search_reused_full_data_hyperparameters"] is False

    # 두 생성기 각각에서 목표 FAKE가 Train·Validation에 없는지 확인한다.
    for target in TARGETS:
        filtered_val = manifest.loc[
            manifest["split"].eq("val")
            & (manifest["label_id"].eq(0) | manifest["generator"].ne(target))
        ]
        filtered_train = manifest.loc[
            manifest["split"].eq("train")
            & (manifest["label_id"].eq(0) | manifest["generator"].ne(target))
        ]
        assert (
            not filtered_train.loc[filtered_train.label_id.eq(1), "generator"]
            .eq(target)
            .any()
        )
        assert (
            not filtered_val.loc[filtered_val.label_id.eq(1), "generator"]
            .eq(target)
            .any()
        )
        expected = manifest.loc[
            manifest["split"].eq("test")
            & (manifest["label_id"].eq(0) | manifest["generator"].eq(target))
        ]
        assert len(expected) == (270 if target == "musicgen" else 279)
        assert expected.loc[expected.label_id.eq(0), "track_sample_id"].nunique() == 45
        assert expected.loc[expected.label_id.eq(1), "track_sample_id"].nunique() == (
            45 if target == "musicgen" else 48
        )
        expected_fake_groups = expected.loc[
            expected.label_id.eq(1), "original_audio"
        ].nunique()
        assert expected_fake_groups == (45 if target == "musicgen" else 24)
        c = cohort.loc[cohort.holdout_generator.eq(target)].iloc[0]
        assert c["primary_segments"] == len(expected)
        assert c["primary_fake_original_audio"] == expected_fake_groups
        assert c["paired_real_tracks"] == expected_fake_groups

        for model in MODELS:
            candidates = tuning.loc[
                tuning.holdout_generator.eq(target) & tuning.model.eq(model)
            ].copy()
            n_candidates = selected["candidate_counts_per_holdout"][model]
            assert len(candidates) == n_candidates
            assert candidates["candidate_order"].tolist() == list(range(n_candidates))
            best = candidates.sort_values(
                ["track_eer", "track_roc_auc", "candidate_order"],
                ascending=[True, False, True],
                kind="stable",
            ).iloc[0]
            choice = selected["models"][target][model]
            assert (
                int(best["candidate_order"]) == choice["candidate"]["candidate_order"]
            )
            for field in ("track_eer", "track_roc_auc", "C"):
                near(
                    best[field], choice["candidate"][field], f"{target}/{model}/{field}"
                )
            if model == "Frozen MERT + LR":
                assert int(best["layer"]) == choice["candidate"]["layer"]
            if model == "RBF-SVM":
                assert str(best["gamma"]) == str(choice["candidate"]["gamma"])
            assert sha256(Path(choice["checkpoint"])) == choice["checkpoint_sha256"]
            assert choice["candidate"]["train_segments"] == len(filtered_train)
            assert choice["candidate"]["val_segments"] == len(filtered_val)

            for level in ("segment", "track"):
                raw = scores.loc[
                    scores.holdout_generator.eq(target)
                    & scores.model.eq(model)
                    & scores.level.eq(level)
                ].copy()
                assert len(raw) == (
                    len(expected)
                    if level == "segment"
                    else expected.track_sample_id.nunique()
                )
                assert np.isfinite(raw["score"]).all()
                near(
                    raw["threshold"].iloc[0],
                    choice["thresholds"][level],
                    "raw threshold",
                )
                assert raw["threshold"].nunique() == 1
                if level == "segment":
                    mapped = raw.set_index("segment_id")
                    assert set(mapped.index) == set(expected["segment_id"])
                    expected_mapped = expected.set_index("segment_id")
                    for a, b in (
                        ("track_id", "track_sample_id"),
                        ("original_audio_id", "original_audio"),
                        ("label", "label_id"),
                    ):
                        assert (
                            mapped[a]
                            .sort_index()
                            .equals(expected_mapped[b].sort_index())
                        )
                    assert set(mapped["track_id"]) == set(expected["track_sample_id"])
                else:
                    seg = scores.loc[
                        scores.holdout_generator.eq(target)
                        & scores.model.eq(model)
                        & scores.level.eq("segment")
                    ]
                    means = seg.groupby("track_id")["score"].mean()
                    for track_id, score in raw.set_index("track_id")["score"].items():
                        near(score, means[track_id], "track mean score")
                    assert set(raw["track_id"]) == set(expected["track_sample_id"])
                assert (
                    raw["prediction"] == (raw["score"] >= raw["threshold"]).astype(int)
                ).all()
                report = metrics.loc[
                    metrics.holdout_generator.eq(target)
                    & metrics.model.eq(model)
                    & metrics.level.eq(level)
                ].iloc[0]
                near(
                    report["validation_segment_threshold"],
                    choice["thresholds"]["segment"],
                    "val segment threshold",
                )
                near(
                    report["validation_track_threshold"],
                    choice["thresholds"]["track"],
                    "val track threshold",
                )
                for key, value in independent_metrics(
                    raw, choice["thresholds"][level]
                ).items():
                    if key == "confusion_matrix":
                        assert json.loads(report[key]) == value
                    elif isinstance(value, int):
                        assert report[key] == value
                    else:
                        near(report[key], value, f"{target}/{model}/{level}/{key}")
                paired_report = paired.loc[
                    paired.holdout_generator.eq(target)
                    & paired.model.eq(model)
                    & paired.level.eq(level)
                ].iloc[0]
                fake_groups = set(
                    expected.loc[expected.label_id.eq(1), "original_audio"]
                )
                paired_raw = raw.loc[raw.original_audio_id.isin(fake_groups)]
                for key, value in independent_metrics(
                    paired_raw, choice["thresholds"][level]
                ).items():
                    if key == "confusion_matrix":
                        assert json.loads(paired_report[key]) == value
                    elif isinstance(value, int):
                        assert paired_report[key] == value
                    else:
                        near(
                            paired_report[key],
                            value,
                            f"paired/{target}/{model}/{level}/{key}",
                        )
    print(
        f"PASS {RUN_ID}: 172 candidates, six checkpoints, 12 primary and 12 paired metric rows"
    )
    print(f"protocol_sha256={protocol_hash}")
    print(f"tuning_sha256={sha256(RESULT / 'tuning.csv')}")
    print(f"selected_configs_sha256={sha256(RESULT / 'selected_configs.json')}")


if __name__ == "__main__":
    main()
