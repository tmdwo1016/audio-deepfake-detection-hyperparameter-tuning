"""Draw four model-specific Baseline/Optimized comparisons from saved Test CSV.

This script reads existing metrics only. It does not load models or evaluate audio.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/model_tuning/optimized_test_results.csv"
OUTPUT = ROOT / "results/model_tuning/model_comparisons"
MODELS = {
    "LogisticRegression": ("Logistic Regression", "logistic_regression"),
    "RBF-SVM": ("RBF-SVM", "rbf_svm"),
    "Log-Mel CNN": ("Log-Mel CNN", "logmel_cnn"),
    "Frozen MERT + LR": ("Frozen MERT + LR", "frozen_mert_lr"),
}
METRICS = (
    ("eer", "EER", False),
    ("roc_auc", "ROC-AUC", True),
    ("real_fpr", "REAL FPR", False),
    ("fake_miss_rate", "FAKE miss", False),
)


def main() -> None:
    # 결과 CSV의 Track 행만 사용한다. 그림에 필요한 값은 기존 Test 결과 그대로다.
    results = pd.read_csv(SOURCE)
    tracks = results.loc[results["level"].eq("track")]
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for model, (title, stem) in MODELS.items():
        rows = tracks.loc[tracks["model"].eq(model)].set_index("variant")
        if set(rows.index) != {"baseline", "optimized"}:
            raise ValueError(f"Expected one Baseline and one Optimized Track row: {model}")

        baseline = rows.loc["baseline"]
        optimized = rows.loc["optimized"]
        if (baseline["n"], optimized["n"]) != (539, 539):
            raise ValueError(f"Unexpected Test cohort size: {model}")

        fig, ax = plt.subplots(figsize=(10.8, 5.0), dpi=180)
        fig.patch.set_facecolor("#f8fafc")
        ax.set_facecolor("#f8fafc")
        positions = np.arange(len(METRICS))[::-1]

        for y, (column, label, higher_is_better) in zip(positions, METRICS):
            before = float(baseline[column])
            after = float(optimized[column])
            delta_pp = (after - before) * 100
            improvement = delta_pp > 0 if higher_is_better else delta_pp < 0
            color = "#197d71" if improvement else "#d27a37"
            if abs(delta_pp) < 0.00005:
                color = "#7b8794"

            # 0을 기준으로 한 막대는 변화량만 표현한다. 실제 전후 값은 오른쪽에 적는다.
            ax.barh(y, delta_pp, height=0.48, color=color, zorder=3)
            # 큰 변화량은 막대 안에 적어 오른쪽 실제 값과 겹치지 않게 한다.
            inside = abs(delta_pp) >= 7.5
            offset = (-0.28 if delta_pp >= 0 else 0.28) if inside else (0.28 if delta_pp >= 0 else -0.28)
            ax.text(
                delta_pp + offset, y, f"{delta_pp:+.2f} pp",
                ha=("right" if delta_pp >= 0 else "left") if inside else ("left" if delta_pp >= 0 else "right"),
                va="center", fontsize=10.5, fontweight="semibold",
                color="white" if inside else "#263442",
            )
            ax.text(
                1.04, y, f"{before:.4f}  →  {after:.4f}",
                transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=10.5, color="#263442", clip_on=False,
            )

        ax.axvline(0, color="#334155", linewidth=1.2, zorder=2)
        ax.set_xlim(-10.5, 10.5)
        ax.set_ylim(-0.7, 3.7)
        ax.set_yticks(positions, [m[1] for m in METRICS], fontsize=11)
        ax.set_xticks([-10, -5, 0, 5, 10])
        ax.set_xlabel("Change from Baseline to Optimized (percentage points)", fontsize=10)
        ax.grid(axis="x", color="#dbe3ea", linewidth=0.8, zorder=1)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis="both", length=0, colors="#334155")
        fig.subplots_adjust(left=0.19, right=0.72, top=0.72, bottom=0.20)

        fig.text(0.075, 0.90, f"{title}  |  Baseline vs Optimized",
                 fontsize=17, fontweight="bold", color="#15283c")
        fig.text(0.075, 0.83, "Track-level Test · 539 tracks (45 REAL, 494 FAKE)",
                 fontsize=10.5, color="#526476")
        fig.text(0.78, 0.745, "Baseline  →  Optimized",
                 fontsize=10, fontweight="semibold", color="#526476")
        fig.text(0.075, 0.07,
                 "Teal: improvement   Orange: worse   Grey: unchanged. "
                 "Error rates use each variant's Validation threshold.",
                 fontsize=9.5, color="#526476")
        path = OUTPUT / f"{stem}_baseline_vs_optimized.png"
        fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
