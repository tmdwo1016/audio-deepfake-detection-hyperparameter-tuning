"""Audit the completed fixed-transfer CNN protocol and Test score arithmetic."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
# 파일 경로로 직접 실행해도 프로젝트 패키지를 가져올 수 있게 한다.
sys.path.insert(0, str(ROOT))

from src.cnn_unseen_fixed import (
    FixedTransferCNNExperiment,
    HOLDOUTS,
    VARIANT,
)
from src.cnn_unseen_revised import id_hash, sha256_file  # noqa: E402
from src.modeling_evaluation import evaluate_binary_predictions  # noqa: E402


def close(a: float, b: float) -> bool:
    return bool(np.isclose(a, b, rtol=1e-9, atol=1e-9))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    ex = FixedTransferCNNExperiment(ROOT, args.run_id)
    # 동결된 선택과 실제 저장 checkpoint를 먼저 연결해 실행 출처를 확인한다.
    frozen = json.loads((ex.result_dir / "selections_frozen.json").read_text())
    primary = pd.read_csv(ex.result_dir / "primary_test_metrics.csv")
    if len(primary) != 4 or set(primary.variant) != {VARIANT}:
        raise AssertionError("Expected exactly two holdouts × two primary levels")
    for holdout in HOLDOUTS:
        train, val = ex.filtered_data(holdout)
        item = frozen[holdout]
        if item["train_segment_sha256"] != id_hash(train) or item[
            "val_segment_sha256"
        ] != id_hash(val):
            raise AssertionError(f"{holdout} split inclusion changed")
        if sha256_file(Path(item["checkpoint"])) != item["checkpoint_sha256"]:
            raise AssertionError(f"{holdout} checkpoint changed after freeze")
        checkpoint = torch.load(
            item["checkpoint"], map_location="cpu", weights_only=False
        )
        if (
            checkpoint["config"] != ex.fixed_config
            or checkpoint["protocol_sha256"] != ex.protocol_hash
        ):
            raise AssertionError(f"{holdout} checkpoint config/protocol changed")
        if checkpoint["train_segment_sha256"] != id_hash(train) or checkpoint[
            "val_segment_sha256"
        ] != id_hash(val):
            raise AssertionError(f"{holdout} checkpoint inclusion changed")
        hist = pd.read_csv(ex.result_dir / holdout / "epoch_history.csv")
        # epoch 재선택 규칙을 저장된 Validation 기록에 독립적으로 적용한다.
        best = min(
            hist.itertuples(),
            key=lambda r: (r.val_track_eer, -r.val_track_roc_auc, r.epoch),
        )
        if int(best.epoch) != item["trial"]["best_epoch"]:
            raise AssertionError(f"{holdout} selected epoch is not Validation best")
        val_scores = pd.read_csv(item["validation_scores"])
        if set(val_scores.segment_id) != set(val.segment_id):
            raise AssertionError(f"{holdout} Validation score IDs differ")
        val_report = evaluate_binary_predictions(val_scores)
        for level in ("segment", "track"):
            if not close(val_report["thresholds"][level], item["thresholds"][level]):
                raise AssertionError(f"{holdout} {level} Validation threshold differs")
        cohort = ex._test_frame(holdout)
        scores = pd.read_csv(
            ex.result_dir / holdout / f"{holdout}_test_segment_scores.csv"
        )
        if len(scores) != len(cohort) or set(scores.segment_id) != set(
            cohort.segment_id
        ):
            raise AssertionError(f"{holdout} Test IDs differ from predeclared cohort")
        # Test의 예측값으로 임계값을 새로 고르지 않고 동결값을 재사용한다.
        report = evaluate_binary_predictions(scores, item["thresholds"])
        for level in ("segment", "track"):
            row = primary[
                (primary.holdout_generator == holdout) & (primary.level == level)
            ].iloc[0]
            for key in (
                "roc_auc",
                "ap_fake",
                "ap_real",
                "eer",
                "balanced_accuracy",
                "macro_f1",
                "real_fpr",
                "fake_miss_rate",
                "hter",
                "threshold",
            ):
                if not close(float(row[key]), float(report[level][key])):
                    raise AssertionError(f"{holdout} {level} {key} differs")
            if json.loads(row.confusion_matrix) != report[level]["confusion_matrix"]:
                raise AssertionError(f"{holdout} {level} confusion matrix differs")
    nb = nbformat.read(ROOT / "27_cnn_unseen_fixed_transfer.ipynb", as_version=4)
    nbformat.validate(nb)
    if any(
        "실행 후" in cell.source for cell in nb.cells if cell.cell_type == "markdown"
    ):
        raise AssertionError("Measured interpretation is incomplete")
    for cell in nb.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            if cell.execution_count is None or not cell.outputs:
                raise AssertionError("Notebook code output missing")
    print(
        "AUDIT_OK: fresh fixed settings, filtered Train/Validation, chosen epochs, "
        "frozen thresholds, exact Test IDs, shared metrics, notebook",
        flush=True,
    )


if __name__ == "__main__":
    main()
