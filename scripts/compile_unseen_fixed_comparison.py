"""Join independently trained fixed-transfer holdout results after ID checks."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLASSICAL = (
    ROOT
    / "results/unseen_fixed_transfer_classical_mert/unseen_fixed_classical_mert_20260919T173138Z"
)
DEFAULT_CNN = (
    ROOT / "results/cnn_unseen_fixed_transfer/cnn_unseen_fixed_20260919T173228Z"
)
DEFAULT_OUTPUT = (
    ROOT / "results/model_robustness/unseen_fixed_four_model_comparison.csv"
)
MODELS = ("LogisticRegression", "RBF-SVM", "Log-Mel CNN", "Frozen MERT + LR")
TARGETS = ("musicgen", "udio")
LEVELS = ("segment", "track")


# 각 모델군의 primary 지표·원점수를 함께 읽어 같은 실행인지 확인한다.
def read_pair(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the metric and raw-score files from one frozen experiment."""
    metrics = pd.read_csv(folder / "primary_test_metrics.csv")
    scores = pd.read_csv(folder / "primary_test_scores.csv")
    if (
        set(metrics["run_id"]) != set(scores["run_id"])
        or len(set(metrics["run_id"])) != 1
    ):
        raise ValueError(f"Run IDs differ within {folder}")
    return metrics, scores


# 네 모델의 같은 holdout·평가 수준만 한 표에서 비교한다.
# 입력: 독립 실행한 classical/MERT와 CNN 결과 폴더.
# 출력: 동일 Test ID와 임계값을 확인한 네 모델 비교표.
def compile_results(classical_dir: Path, cnn_dir: Path) -> pd.DataFrame:
    """Require the same Test IDs and labels for every target and level."""
    classical, old_scores = read_pair(classical_dir)
    cnn, cnn_scores = read_pair(cnn_dir)
    common_columns = [column for column in classical.columns if column in cnn.columns]
    metrics = pd.concat(
        [classical[common_columns], cnn[common_columns]], ignore_index=True
    )
    scores = pd.concat([old_scores, cnn_scores], ignore_index=True)
    if (
        len(metrics) != len(TARGETS) * len(MODELS) * len(LEVELS)
        or set(metrics["variant"]) != {"fixed_hyperparameter_transfer"}
        or set(metrics["cohort"]) != {"primary"}
        or set(metrics["split"]) != {"test"}
    ):
        raise ValueError("Incomplete or mixed primary fixed-transfer metrics")
    manifest = pd.read_csv(ROOT / "data/metadata/segment_manifest_10s.csv")
    for target in TARGETS:
        expected = manifest.loc[
            manifest["split"].eq("test")
            & (manifest["label_id"].eq(0) | manifest["generator"].eq(target))
        ]
        expected_ids = set(expected["segment_id"])
        if expected["track_sample_id"].nunique() != (
            90 if target == "musicgen" else 93
        ) or len(expected) != (270 if target == "musicgen" else 279):
            raise ValueError(f"Original {target} Test cohort changed")
        for model in MODELS:
            pair = metrics.loc[
                metrics["holdout_generator"].eq(target) & metrics["model"].eq(model)
            ]
            if set(pair["level"]) != set(LEVELS) or len(pair) != 2:
                raise ValueError(f"Missing {target}/{model} Segment or Track row")
            segment = scores.loc[
                scores["holdout_generator"].eq(target)
                & scores["model"].eq(model)
                & scores["level"].eq("segment")
            ]
            track = scores.loc[
                scores["holdout_generator"].eq(target)
                & scores["model"].eq(model)
                & scores["level"].eq("track")
            ]
            if set(segment["segment_id"]) != expected_ids or len(segment) != len(
                expected
            ):
                raise ValueError(f"{target}/{model} Segment IDs differ")
            if (
                set(track["track_id"]) != set(expected["track_sample_id"])
                or len(track) != expected["track_sample_id"].nunique()
            ):
                raise ValueError(f"{target}/{model} Track IDs differ")
            reference = expected[
                ["segment_id", "track_sample_id", "original_audio", "label_id"]
            ]
            joined = segment.merge(reference, on="segment_id", validate="one_to_one")
            if (
                not joined["track_id"].equals(joined["track_sample_id"])
                or not joined["original_audio_id"].equals(joined["original_audio"])
                or not joined["label"].equals(joined["label_id"])
            ):
                raise ValueError(f"{target}/{model} Test metadata differs")
            means = segment.groupby("track_id")["score"].mean()
            stored = track.set_index("track_id")["score"]
            if not np.allclose(means.loc[stored.index], stored, rtol=0, atol=1e-12):
                raise ValueError(f"{target}/{model} Track scores are not Segment means")
            for level, frame in (("segment", segment), ("track", track)):
                row = pair.loc[pair["level"].eq(level)].iloc[0]
                if (
                    int(row["n"]) != len(frame)
                    or int(row["n_real"]) != int(frame["label"].eq(0).sum())
                    or int(row["n_fake"]) != int(frame["label"].eq(1).sum())
                    or not np.allclose(frame["threshold"], float(row["threshold"]))
                ):
                    raise ValueError(
                        f"{target}/{model}/{level} count or threshold differs"
                    )
    order = {model: index for index, model in enumerate(MODELS)}
    metrics["_model_order"] = metrics["model"].map(order)
    metrics["_target_order"] = metrics["holdout_generator"].map(
        {target: index for index, target in enumerate(TARGETS)}
    )
    metrics["_level_order"] = metrics["level"].map({"track": 0, "segment": 1})
    return (
        metrics.sort_values(["_target_order", "_level_order", "_model_order"])
        .drop(columns=["_model_order", "_target_order", "_level_order"])
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classical", type=Path, default=DEFAULT_CLASSICAL)
    parser.add_argument("--cnn", type=Path, default=DEFAULT_CNN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    compiled = compile_results(args.classical, args.cnn)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # 검증된 공통 비교표만 별도 CSV로 저장한다.
    compiled.to_csv(args.output, index=False, encoding="utf-8-sig")
    track = compiled.loc[compiled["level"].eq("track")]
    for metric, label in (("roc_auc", "Track ROC-AUC"), ("eer", "Track EER")):
        fig, ax = plt.subplots(figsize=(10, 5))
        centers = np.arange(len(MODELS))
        for offset, target in ((-0.19, "musicgen"), (0.19, "udio")):
            subset = track.loc[track["holdout_generator"].eq(target)].set_index("model")
            ax.bar(
                centers + offset,
                subset.loc[list(MODELS), metric],
                width=0.36,
                label=target,
            )
        ax.set_xticks(centers, ("LR", "RBF-SVM", "Log-Mel CNN", "Frozen MERT + LR"))
        ax.set_ylabel(label)
        ax.set_title(f"Fixed-hyperparameter transfer: {label}")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output.with_name(f"unseen_fixed_{metric}.png"), dpi=160)
        plt.close(fig)
    print(f"PASS {len(compiled)} fixed-transfer rows: {args.output}")


if __name__ == "__main__":
    main()
