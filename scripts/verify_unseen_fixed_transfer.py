"""Read-only independent audit of the fixed-hyperparameter transfer run."""

# ruff: noqa: E402

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# 파일 경로로 실행할 때도 결과 검증 모듈을 찾도록 루트를 등록한다.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modeling_evaluation import select_validation_threshold  # noqa: E402
from scripts.verify_unseen_classical_mert import (
    independent_metrics,
    near,
    sha256,
)

RUN_ID = "unseen_fixed_classical_mert_20260919T173138Z"
RESULT = ROOT / "results/unseen_fixed_transfer_classical_mert" / RUN_ID
MODELS = ("LogisticRegression", "RBF-SVM", "Frozen MERT + LR")
TARGETS = ("musicgen", "udio")


def main() -> None:
    manifest = pd.read_csv(ROOT / "data/metadata/segment_manifest_10s.csv")
    # 새 고정 설정 실행의 선택·실행 메타데이터를 함께 대조한다.
    selected = json.loads((RESULT / "fixed_selections.json").read_text())
    meta = json.loads((RESULT / "run_metadata.json").read_text())
    primary = pd.read_csv(RESULT / "primary_test_metrics.csv")
    paired = pd.read_csv(RESULT / "paired_source_test_metrics.csv")
    scores = pd.read_csv(RESULT / "primary_test_scores.csv")
    cohorts = pd.read_csv(RESULT / "test_cohorts.csv")
    assert selected["run_id"] == meta["run_id"] == RUN_ID
    assert selected["variant"] == meta["variant"] == "fixed_hyperparameter_transfer"
    assert (
        selected["protocol_sha256"]
        == meta["protocol_sha256"]
        == sha256(ROOT / "docs/UNSEEN_FIXED_PROTOCOL.md")
    )
    assert (
        selected["protocol_sha256"]
        == "098fa65192f65fddbf7d65757eb5c6b5c2b1149f79178c27746b4e328ee54a5e"
    )
    assert (
        selected["manifest_sha256"]
        == meta["manifest_sha256"]
        == sha256(ROOT / "data/metadata/segment_manifest_10s.csv")
    )
    assert sha256(RESULT / "fixed_selections.json") == meta["fixed_selections_sha256"]
    # 하이퍼파라미터는 Main 값이고 임계값만 필터된 Validation 값이다.
    assert meta["hyperparameters_reselected_on_filtered_val"] is False
    assert meta["thresholds_selected_on_filtered_val"] is True
    assert len(primary) == len(paired) == 12
    assert set(scores["run_id"]) == {RUN_ID}
    assert set(scores["variant"]) == {"fixed_hyperparameter_transfer"}
    assert set(scores["cohort"]) == {"primary"}
    assert set(scores["split"]) == {"test"}

    source_classical = ROOT / "results/model_tuning/classical_best_configs.json"
    source_mert = (
        ROOT / "results/model_tuning/mert_best_config_mert_binary_20260919T110741Z.json"
    )
    assert (
        meta["source_config_sha256"]
        == selected["source_config_sha256"]
        == {"classical": sha256(source_classical), "mert": sha256(source_mert)}
    )
    # 원래 Main의 LR/SVM/MERT 설정이 정확히 보존됐는지 확인한다.
    assert selected["main_best"] == {
        "LogisticRegression": {"C": 0.01},
        "RBF-SVM": {"C": 10.0, "gamma": 0.001},
        "Frozen MERT + LR": {"layer": 4, "C": 0.01},
    }
    columns = json.loads(
        (ROOT / "results/model_tuning/classical_run_metadata.json").read_text()
    )["feature_columns"]
    features = pd.read_csv(
        ROOT / "data/processed/features/handcrafted_features_10s.csv"
    )
    features = features.set_index("segment_id")
    mert_path = (
        ROOT
        / "data/processed/mert/mert95m_binary_train_val_validframe_12af15fef9d0.npy"
    )
    mert_all = np.load(mert_path, mmap_mode="r")
    train_val = manifest.loc[manifest.split.isin(("train", "val"))].reset_index(
        drop=True
    )
    mert_index = pd.Index(train_val["segment_id"])

    for target in TARGETS:
        train = manifest.loc[
            manifest.split.eq("train")
            & (manifest.label_id.eq(0) | manifest.generator.ne(target))
        ]
        val = manifest.loc[
            manifest.split.eq("val")
            & (manifest.label_id.eq(0) | manifest.generator.ne(target))
        ]
        assert not train.loc[train.label_id.eq(1), "generator"].eq(target).any()
        assert not val.loc[val.label_id.eq(1), "generator"].eq(target).any()
        expected = manifest.loc[
            manifest.split.eq("test")
            & (manifest.label_id.eq(0) | manifest.generator.eq(target))
        ]
        assert len(expected) == (270 if target == "musicgen" else 279)
        cohort = cohorts.loc[cohorts.holdout_generator.eq(target)].iloc[0]
        assert cohort["primary_segments"] == len(expected)
        assert cohort["primary_real_tracks"] == 45
        assert cohort["primary_fake_tracks"] == (45 if target == "musicgen" else 48)
        assert cohort["primary_fake_original_audio"] == (
            45 if target == "musicgen" else 24
        )
        fake_groups = set(expected.loc[expected.label_id.eq(1), "original_audio"])
        assert cohort["paired_real_tracks"] == len(fake_groups)

        for model in MODELS:
            choice = selected["selected"][target][model]
            assert choice["train_segments"] == len(train)
            assert choice["val_segments"] == len(val)
            assert sha256(Path(choice["checkpoint"])) == choice["checkpoint_sha256"]
            assert (
                choice["checkpoint_sha256"] == meta["checkpoint_sha256"][target][model]
            )
            checkpoint = joblib.load(choice["checkpoint"])
            if model in MODELS[:2]:
                clf = checkpoint.named_steps["classifier"]
                assert clf.class_weight == "balanced" and list(clf.classes_) == [0, 1]
                assert clf.C == (0.01 if model == "LogisticRegression" else 10.0)
                if model == "RBF-SVM":
                    assert clf.gamma == 0.001 and clf.probability is False
                expected_mean = (
                    features.loc[train["segment_id"], columns]
                    .to_numpy(float)
                    .mean(axis=0)
                )
                assert np.allclose(
                    checkpoint.named_steps["scaler"].mean_,
                    expected_mean,
                    atol=1e-9,
                    rtol=1e-9,
                )
            else:
                assert checkpoint["layer"] == 4 and checkpoint["C"] == 0.01
                assert checkpoint["classifier"].class_weight == "balanced"
                assert list(checkpoint["classifier"].classes_) == [0, 1]
                positions = mert_index.get_indexer(train["segment_id"])
                assert (positions >= 0).all()
                expected_mean = np.asarray(
                    mert_all[positions, 4, :], dtype=np.float32
                ).mean(axis=0)
                assert np.allclose(
                    checkpoint["scaler"].mean_, expected_mean, atol=2e-5, rtol=1e-6
                )

            safe = (
                model.lower().replace(" ", "_").replace("+", "plus").replace("-", "_")
            )
            for level in ("segment", "track"):
                val_raw = pd.read_csv(
                    RESULT / f"{target}_{safe}_{level}_val_scores.csv"
                )
                assert val_raw["threshold"].nunique() == 1
                near(
                    val_raw["threshold"].iloc[0],
                    choice["thresholds"][level],
                    "Val threshold",
                )
                expected_ids = set(
                    val["segment_id"] if level == "segment" else val["track_sample_id"]
                )
                actual_ids = set(
                    val_raw["segment_id"] if level == "segment" else val_raw["track_id"]
                )
                assert actual_ids == expected_ids
                near(
                    # 저장된 Validation 점수에서 임계값을 재계산해 Test 누수를 점검한다.
                    select_validation_threshold(val_raw["label"], val_raw["score"]),
                    choice["thresholds"][level],
                    "reselected Validation threshold",
                )

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
                    "Test threshold",
                )
                assert raw["threshold"].nunique() == 1
                if level == "segment":
                    actual = raw.set_index("segment_id")
                    ref = expected.set_index("segment_id")
                    assert set(actual.index) == set(ref.index)
                    for a, b in (
                        ("track_id", "track_sample_id"),
                        ("original_audio_id", "original_audio"),
                        ("label", "label_id"),
                    ):
                        assert actual[a].sort_index().equals(ref[b].sort_index())
                else:
                    segment = scores.loc[
                        scores.holdout_generator.eq(target)
                        & scores.model.eq(model)
                        & scores.level.eq("segment")
                    ]
                    means = segment.groupby("track_id")["score"].mean()
                    for track_id, score in raw.set_index("track_id")["score"].items():
                        near(score, means[track_id], "Track mean")
                    assert set(raw["track_id"]) == set(expected["track_sample_id"])
                report = primary.loc[
                    primary.holdout_generator.eq(target)
                    & primary.model.eq(model)
                    & primary.level.eq(level)
                ].iloc[0]
                near(
                    report["validation_segment_threshold"],
                    choice["thresholds"]["segment"],
                    "segment threshold",
                )
                near(
                    report["validation_track_threshold"],
                    choice["thresholds"]["track"],
                    "track threshold",
                )
                for field, value in independent_metrics(
                    raw, choice["thresholds"][level]
                ).items():
                    if field == "confusion_matrix":
                        assert json.loads(report[field]) == value
                    elif isinstance(value, int):
                        assert report[field] == value
                    else:
                        near(report[field], value, f"{target}/{model}/{level}/{field}")
                paired_raw = raw.loc[raw.original_audio_id.isin(fake_groups)]
                paired_report = paired.loc[
                    paired.holdout_generator.eq(target)
                    & paired.model.eq(model)
                    & paired.level.eq(level)
                ].iloc[0]
                for field, value in independent_metrics(
                    paired_raw, choice["thresholds"][level]
                ).items():
                    if field == "confusion_matrix":
                        assert json.loads(paired_report[field]) == value
                    elif isinstance(value, int):
                        assert paired_report[field] == value
                    else:
                        near(
                            paired_report[field],
                            value,
                            f"paired/{target}/{model}/{level}/{field}",
                        )
    print(
        f"PASS {RUN_ID}: six fresh fixed fits, filtered scalers, Val thresholds, 24 Test metric rows"
    )


if __name__ == "__main__":
    main()
