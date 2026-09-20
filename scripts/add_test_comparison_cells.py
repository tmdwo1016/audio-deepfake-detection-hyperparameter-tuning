"""Append executed Test comparison cells using previously saved final results.

Only the new plotting cells run. The models and Test predictions are never
recomputed; the original notebook cells and their outputs stay unchanged.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from textwrap import dedent

import nbformat

from add_validation_comparison_cells import (
    PREAMBLE,
    ROOT,
    execute_new_cells,
)


MARKER = "저장된 공동 Test 비교"


def test_code(model_name: str, figure_name: str, run_config: str) -> str:
    """Create one self-contained cell for a model's saved Track Test rows."""
    return PREAMBLE + dedent(f"""\
        # Validation 후보 그림과 공동 Test 그림은 폴더도 구분해 저장한다.
        figure_dir = result_dir / "test_comparisons"
        figure_dir.mkdir(parents=True, exist_ok=True)

        # 최종 공동 Test CSV에서 이 모델의 곡 단위 두 행만 읽는다.
        model_name = {model_name!r}
        final_results = pd.read_csv(result_dir / "optimized_test_results.csv")
        rows = final_results.loc[
            (final_results["model"] == model_name)
            & (final_results["split"] == "test")
            & (final_results["level"] == "track")
        ].copy()
        assert len(rows) == 2 and set(rows["variant"]) == {{"baseline", "optimized"}}
        expected_run_id = {run_config}
        assert rows["run_id"].nunique() == 1 and rows["run_id"].iloc[0] == expected_run_id
        assert rows["n"].eq(539).all() and rows["n_real"].eq(45).all() and rows["n_fake"].eq(494).all()
        baseline = rows.set_index("variant").loc["baseline"]
        selected = rows.set_index("variant").loc["optimized"]

        # EER과 AUC는 Test 점수에서 계산됐다. 분류 오류율에는 각 설정의 Validation 임계값을 썼다.
        summary = pd.DataFrame({{
            "설정": ["Baseline", "Selected"],
            "Validation threshold": [baseline["threshold"], selected["threshold"]],
            "ROC-AUC ↑": [baseline["roc_auc"], selected["roc_auc"]],
            "EER ↓": [baseline["eer"], selected["eer"]],
            "Macro-F1 ↑": [baseline["macro_f1"], selected["macro_f1"]],
            "REAL FPR ↓": [baseline["real_fpr"], selected["real_fpr"]],
            "FAKE miss ↓": [baseline["fake_miss_rate"], selected["fake_miss_rate"]],
        }})
        display(summary.round(4))

        # 왼쪽은 앞의 Validation 비교와 같은 세 지표, 오른쪽은 오류 방향을 보여 준다.
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        main_metrics = [("ROC-AUC", "roc_auc"), ("EER", "eer"), ("Macro-F1", "macro_f1")]
        error_metrics = [("REAL FPR", "real_fpr"), ("FAKE miss", "fake_miss_rate")]
        for ax, metrics, title in [
            (axes[0], main_metrics, "Same Test tracks: main metrics"),
            (axes[1], error_metrics, "Errors at Validation thresholds"),
        ]:
            positions = np.arange(len(metrics))
            baseline_values = [baseline[column] for _, column in metrics]
            selected_values = [selected[column] for _, column in metrics]
            baseline_bars = ax.bar(positions - 0.18, baseline_values, 0.36, label="Baseline", color="#3973a6")
            selected_bars = ax.bar(positions + 0.18, selected_values, 0.36, label="Selected", color="#ef8b38")
            ax.bar_label(baseline_bars, fmt="%.3f", padding=2, fontsize=8)
            ax.bar_label(selected_bars, fmt="%.3f", padding=2, fontsize=8)
            ax.set_xticks(positions, [label for label, _ in metrics])
            ax.set_ylim(0, 1.10 if ax is axes[0] else max(baseline_values + selected_values) * 1.3 + 0.01)
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.2)
        axes[0].legend(loc="upper center", ncol=2)
        fig.tight_layout()
        figure_path = figure_dir / {figure_name!r}
        fig.savefig(figure_path, dpi=160, bbox_inches="tight")
        plt.show()
        print("Saved:", figure_path.relative_to(project_root))
    """)


def append_test_sections(notebook_name: str, sections: list[tuple[str, str]]) -> None:
    path = ROOT / "archive_notebooks" / notebook_name
    notebook = nbformat.read(path, as_version=4)
    positions = [index for index, cell in enumerate(notebook.cells) if MARKER in cell.source]
    if positions:
        expected_start = len(notebook.cells) - 2 * len(sections)
        if positions[0] != expected_start or len(positions) != len(sections):
            raise RuntimeError(f"Unexpected Test comparison cell layout: {notebook_name}")
        notebook.cells = notebook.cells[:expected_start]
    original_cells = deepcopy(notebook.cells)
    additions = []
    for title, source in sections:
        additions.extend([nbformat.v4.new_markdown_cell(title), nbformat.v4.new_code_cell(source)])
    executed = execute_new_cells(additions)
    assert all(cell.cell_type != "code" or cell.outputs for cell in executed)
    next_count = max(
        (cell.execution_count or 0 for cell in original_cells if cell.cell_type == "code"),
        default=0,
    )
    for cell in executed:
        if cell.cell_type == "code":
            next_count += 1
            cell.execution_count = next_count
    notebook.cells.extend(executed)
    nbformat.validate(notebook)
    assert notebook.cells[: len(original_cells)] == original_cells
    nbformat.write(notebook, path)
    print(f"Added {len(sections)} saved-Test comparison(s): {notebook_name}")


def main() -> None:
    classical_run = 'json.loads((result_dir / "classical_best_configs.json").read_text())["run_id"]'
    cnn_run = 'json.loads((result_dir / "cnn_optimized_selection.json").read_text())["run_id"]'
    mert_run = 'json.loads(next(result_dir.glob("mert_best_config_mert_binary_*.json")).read_text())["run_id"]'
    note = (
        "저장된 공동 Test 결과의 같은 539곡을 비교한다. AUC·EER은 Test 점수에서 계산했고, "
        "Macro-F1·REAL FPR·FAKE miss에는 각 설정의 Validation 임계값을 적용했다. "
        "여기서는 모델이나 임계값을 다시 선택하지 않는다."
    )
    append_test_sections(
        "22_classical_model_tuning.ipynb",
        [
            (
                f"## 10. Logistic Regression — {MARKER}\n\n{note}",
                test_code("LogisticRegression", "logistic_test_comparison.png", classical_run),
            ),
            (
                f"## 11. RBF-SVM — {MARKER}\n\n{note}",
                test_code("RBF-SVM", "svm_test_comparison.png", classical_run),
            ),
        ],
    )
    append_test_sections(
        "13B_logmel_cnn_tuning.ipynb",
        [
            (
                f"## 10. CNN — {MARKER}\n\n{note}",
                test_code("Log-Mel CNN", "cnn_test_comparison.png", cnn_run),
            ),
        ],
    )
    append_test_sections(
        "17B_mert_binary_tuning.ipynb",
        [
            (
                f"## 9. MERT+LR — {MARKER}\n\n{note}",
                test_code("Frozen MERT + LR", "mert_test_comparison.png", mert_run),
            ),
        ],
    )


if __name__ == "__main__":
    main()
