"""Shared REAL=0 / FAKE=1 evaluation for the revised binary experiments.

Use this module on Validation to choose thresholds. Pass those saved thresholds
to the final Test call; Test ROC/EER never determines a prediction threshold.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

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

REQUIRED_COLUMNS = ("segment_id", "track_id", "original_audio_id", "label", "score")


def validate_segment_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Check identifiers, labels and finite scores before any aggregation."""
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing evaluation columns: {sorted(missing)}")
    out = frame.loc[:, REQUIRED_COLUMNS].copy()
    if out.empty:
        raise ValueError("Evaluation frame has no segments")
    if (
        out[["segment_id", "track_id", "original_audio_id", "label", "score"]]
        .isna()
        .any()
        .any()
    ):
        raise ValueError("Evaluation frame contains null IDs, labels or scores")
    if out["segment_id"].duplicated().any():
        duplicates = (
            out.loc[out["segment_id"].duplicated(), "segment_id"].head(5).tolist()
        )
        raise ValueError(f"Duplicate segment_id: {duplicates}")
    # REAL=0·FAKE=1을 강제해야 ROC와 오류율의 양성 방향이 뒤집히지 않는다.
    labels = pd.to_numeric(out["label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("Labels must be numeric REAL=0 or FAKE=1")
    out["label"] = labels.astype(np.int8)
    scores = pd.to_numeric(out["score"], errors="coerce")
    if scores.isna().any() or not np.isfinite(scores.to_numpy(dtype=float)).all():
        raise ValueError("Scores must all be finite numbers")
    out["score"] = scores.astype(float)
    if (out.groupby("track_id", sort=False)["label"].nunique() > 1).any():
        raise ValueError("A track contains both REAL and FAKE labels")
    if (out.groupby("track_id", sort=False)["original_audio_id"].nunique() > 1).any():
        raise ValueError("A track maps to more than one original_audio_id")
    return out


def aggregate_track_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Average segment scores within each track after strict consistency checks."""
    valid = validate_segment_frame(frame)
    # 같은 곡의 여러 10초 구간을 평균한다. 곡별 구간 수로 다시 가중하지 않는다.
    return valid.groupby("track_id", sort=False, as_index=False).agg(
        original_audio_id=("original_audio_id", "first"),
        label=("label", "first"),
        score=("score", "mean"),
        n_segments=("segment_id", "size"),
    )


def _roc_details(y_true: np.ndarray, scores: np.ndarray):
    """Return full finite ROC candidates and linearly interpolated EER."""
    fpr, tpr, thresholds = roc_curve(
        y_true, scores, pos_label=1, drop_intermediate=False
    )
    fnr = 1.0 - tpr
    difference = fpr - fnr
    exact = np.flatnonzero(difference == 0)
    if len(exact):
        eer = float(fpr[exact[0]])
    else:
        crossings = np.flatnonzero(difference[:-1] * difference[1:] < 0)
        if not len(crossings):
            raise RuntimeError("ROC FPR-FNR intersection was not found")
        i = int(crossings[0])
        # EER는 ROC 위 두 점 사이를 보간한 통계량이며 분류 임계값이 아니다.
        weight = -difference[i] / (difference[i + 1] - difference[i])
        eer = float(fpr[i] + weight * (fpr[i + 1] - fpr[i]))
    return fpr, fnr, thresholds, eer


def select_validation_threshold(
    y_true: Sequence[int], scores: Sequence[float]
) -> float:
    """Choose a finite Validation ROC threshold by error gap, HTER, then height."""
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    if y.ndim != 1 or s.ndim != 1 or len(y) != len(s) or not len(y):
        raise ValueError("Labels and scores must be equal-length nonempty vectors")
    if not np.isfinite(s).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be 0/1 and scores finite")
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, fnr, thresholds, _ = _roc_details(y, s)
    candidates = np.flatnonzero(np.isfinite(thresholds))
    if not len(candidates):
        raise RuntimeError("ROC returned no finite threshold candidates")
    # 무한대 ROC 시작점을 빼고 오류 차이 → HTER → 큰 임계값 순으로 선택한다.
    best = min(
        candidates,
        key=lambda i: (abs(fpr[i] - fnr[i]), (fpr[i] + fnr[i]) / 2, -thresholds[i]),
    )
    return float(thresholds[best])


def _metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Calculate ranking and fixed-threshold classification metrics."""
    n_real = int(np.sum(y == 0))
    n_fake = int(np.sum(y == 1))
    result = {
        "n": int(len(y)),
        "n_real": n_real,
        "n_fake": n_fake,
        "real_prevalence": n_real / len(y),
        "fake_prevalence": n_fake / len(y),
        "threshold": float(threshold),
        "reason": None,
    }
    if n_real == 0 or n_fake == 0:
        result.update(
            {
                key: float("nan")
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
                )
            }
        )
        result["reason"] = "only one class is present"
        result["confusion_matrix"] = None
        return result
    if not np.isfinite(threshold):
        raise ValueError("A finite Validation-derived threshold is required")
    # Test에서도 저장된 Validation 임계값을 사용한다. Test EER 임계값은 쓰지 않는다.
    # 점수가 임계값 이상이면 FAKE로 판정한다; 임계값은 Validation에서 결정된다.
    prediction = (scores >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    _, _, _, eer = _roc_details(y, scores)
    real_fpr = float(fp / (tn + fp))
    fake_miss_rate = float(fn / (fn + tp))
    result.update(
        {
            "roc_auc": float(roc_auc_score(y, scores)),
            "ap_fake": float(average_precision_score(y, scores)),
            "ap_real": float(average_precision_score(1 - y, -scores)),
            "eer": eer,
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
            "macro_f1": float(
                f1_score(y, prediction, labels=[0, 1], average="macro", zero_division=0)
            ),
            "real_fpr": real_fpr,
            "fake_miss_rate": fake_miss_rate,
            "hter": (real_fpr + fake_miss_rate) / 2,
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        }
    )
    return result


# 입력: ID·라벨·FAKE 점수 표와 선택적으로 저장된 수준별 임계값.
# 출력: Segment·Track 지표, 적용 임계값, 예측이 붙은 원점수 표.
def evaluate_binary_predictions(
    frame: pd.DataFrame, thresholds: Mapping[str, float] | None = None
) -> dict:
    """Evaluate segment and mean-score track levels with separate thresholds."""
    segments = validate_segment_frame(frame)
    tracks = aggregate_track_scores(segments)
    levels = {"segment": segments, "track": tracks}
    if thresholds is not None and set(thresholds) != set(levels):
        raise ValueError("Pass saved Validation thresholds for segment and track")
    chosen = {}
    result = {}
    for level, scores_frame in levels.items():
        y = scores_frame["label"].to_numpy(dtype=np.int8)
        scores = scores_frame["score"].to_numpy(dtype=float)
        threshold = (
            select_validation_threshold(y, scores)
            if thresholds is None
            else float(thresholds[level])
        )
        chosen[level] = threshold
        result[level] = _metrics(y, scores, threshold)
        scores_frame["prediction"] = (
            (scores >= threshold).astype(np.int8)
            if np.isfinite(threshold)
            else pd.array([pd.NA] * len(scores), dtype="Int8")
        )
    result["thresholds"] = chosen
    result["segment_scores"] = segments
    result["track_scores"] = tracks
    return result


def select_best_candidate(records: pd.DataFrame | Sequence[Mapping]) -> dict:
    """Select by raw Validation track EER, descending AUC, then fixed order."""
    rows = (
        records.to_dict("records")
        if isinstance(records, pd.DataFrame)
        else list(records)
    )
    if not rows:
        raise ValueError("No successful candidates")
    required = {"track_eer", "track_roc_auc", "candidate_order"}
    if any(not required.issubset(row) for row in rows):
        raise ValueError(f"Candidate rows require {sorted(required)}")
    # 반올림 전 Track EER로 고르고 동률일 때 AUC와 사전 후보 순서를 쓴다.
    valid = [row for row in rows if np.isfinite(row["track_eer"])]
    if not valid:
        raise ValueError("No candidate has a finite Validation Track EER")
    return dict(
        min(
            valid,
            key=lambda row: (
                float(row["track_eer"]),
                (
                    -float(row["track_roc_auc"])
                    if np.isfinite(row["track_roc_auc"])
                    else float("inf")
                ),
                int(row["candidate_order"]),
            ),
        )
    )


def assert_group_disjoint(frame: pd.DataFrame, split_col: str = "split") -> None:
    """Stop if an original audio group occurs in more than one split."""
    if "original_audio_id" not in frame or split_col not in frame:
        raise ValueError("Group check needs original_audio_id and split")
    if frame[["original_audio_id", split_col]].isna().any().any():
        raise ValueError("Group check contains missing IDs or split")
    # 같은 원곡이 둘 이상의 split에 있으면 학습·평가 누수가 발생한다.
    overlap = frame.groupby("original_audio_id")[split_col].nunique()
    if (overlap > 1).any():
        raise ValueError(
            f"Original-audio split leakage: {overlap[overlap > 1].index[:5].tolist()}"
        )
