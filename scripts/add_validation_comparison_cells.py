"""Append executed, read-only validation comparison cells to tuning notebooks.

This script reads saved candidate CSV/JSON files. It never trains a model or
opens the Test split. The new cells are executed alone, then appended to the
original notebooks so earlier execution counts and outputs stay intact.
"""

from __future__ import annotations

import base64
from contextlib import redirect_stdout
from copy import deepcopy
from io import BytesIO, StringIO
from pathlib import Path
from textwrap import dedent

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
from IPython import display as ipython_display


ROOT = Path(__file__).resolve().parents[1]
MARKER = "저장된 Validation 후보와 Baseline 비교"

PREAMBLE = dedent("""\
    # 저장된 후보 결과만 읽는다. 이 셀은 모델을 다시 학습하거나 Test를 열지 않는다.
    from pathlib import Path
    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display

    project_root = Path.cwd()
    if not (project_root / "results" / "model_tuning").is_dir():
        project_root = project_root.parent
    result_dir = project_root / "results" / "model_tuning"
    assert result_dir.is_dir(), "프로젝트 루트 또는 archive_notebooks에서 실행하세요."
    figure_dir = result_dir / "validation_comparisons"
    figure_dir.mkdir(parents=True, exist_ok=True)
""")

BAR_PLOT = dedent("""\
    # 같은 Validation Track 지표를 두 설정에서 나란히 비교한다.
    metric_names = ["ROC-AUC", "EER", "Macro-F1"]
    baseline_values = [baseline["track_roc_auc"], baseline["track_eer"], baseline["track_macro_f1"]]
    selected_values = [selected["track_roc_auc"], selected["track_eer"], selected["track_macro_f1"]]
    positions = np.arange(len(metric_names))
    baseline_bars = axes[1].bar(positions - 0.18, baseline_values, 0.36, label="Baseline", color="#3973a6")
    selected_bars = axes[1].bar(positions + 0.18, selected_values, 0.36, label="Selected", color="#ef8b38")
    axes[1].bar_label(baseline_bars, fmt="%.3f", padding=2, fontsize=8)
    axes[1].bar_label(selected_bars, fmt="%.3f", padding=2, fontsize=8)
    axes[1].set_xticks(positions, metric_names)
    axes[1].set_ylim(0, 1.10)
    axes[1].set_title("Baseline vs selected on Validation tracks")
    axes[1].legend(loc="upper center", ncol=2)
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.show()
    print("Saved:", figure_path.relative_to(project_root))
""")


def classical_code(model_name: str) -> str:
    """Build a separate LR or SVM cell with the full candidate overview."""
    csv_name = "logistic_tuning.csv" if model_name == "LogisticRegression" else "svm_tuning.csv"
    image_name = "logistic_validation_comparison.png" if model_name == "LogisticRegression" else "svm_validation_comparison.png"
    model_title = "Logistic Regression" if model_name == "LogisticRegression" else "RBF-SVM"
    model_part = dedent(f"""\
        model_name = {model_name!r}
        candidates = pd.read_csv(result_dir / {csv_name!r}).sort_values("candidate_order")
        baseline_config = json.loads((result_dir / "classical_baseline_configs.json").read_text())
        selected_config = json.loads((result_dir / "classical_best_configs.json").read_text())
        assert candidates["run_id"].nunique() == 1
        assert candidates["run_id"].iloc[0] == baseline_config["run_id"] == selected_config["run_id"]
        baseline_order = baseline_config["models"][model_name]["candidate_order"]
        selected_order = selected_config["models"][model_name]["candidate_order"]
        baseline = candidates.loc[candidates["candidate_order"] == baseline_order].iloc[0]
        selected = candidates.loc[candidates["candidate_order"] == selected_order].iloc[0]
        assert np.isclose(baseline["C"], baseline_config["models"][model_name]["params"]["C"])
        assert np.isclose(selected["C"], selected_config["models"][model_name]["params"]["C"])
    """)
    if model_name == "LogisticRegression":
        specific = dedent("""\
            assert len(candidates) == 5
            # 다섯 C 후보를 전부 보여 주고, 선택 기준인 Track EER를 그린다.
            display(candidates[["C", "track_roc_auc", "track_eer", "val_track_macro_f1"]].round(4))
            summary = pd.DataFrame({
                "설정": ["Baseline", "Selected"],
                "C": [baseline["C"], selected["C"]],
                "Track ROC-AUC ↑": [baseline["track_roc_auc"], selected["track_roc_auc"]],
                "Track EER ↓": [baseline["track_eer"], selected["track_eer"]],
                "Track Macro-F1 ↑": [baseline["val_track_macro_f1"], selected["val_track_macro_f1"]],
            })
            display(summary.round(4))

            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            axes[0].plot(candidates["C"], candidates["track_eer"], "o-", color="#64748b")
            axes[0].scatter([baseline["C"]], [baseline["track_eer"]], s=110, label="Baseline", color="#3973a6", zorder=3)
            axes[0].scatter([selected["C"]], [selected["track_eer"]], s=110, label="Selected", color="#ef8b38", zorder=3)
            axes[0].set_xscale("log")
            axes[0].set_xticks(candidates["C"], [f"{c:g}" for c in candidates["C"]])
            axes[0].set_xlabel("C (smaller means stronger regularization)")
            axes[0].set_ylabel("Validation Track EER ↓")
            axes[0].set_title("All 5 C candidates")
            axes[0].legend()
            axes[0].grid(alpha=0.2)
        """)
    else:
        specific = dedent("""\
            assert len(candidates) == 16
            assert str(baseline["gamma"]) == str(baseline_config["models"][model_name]["params"]["gamma"])
            assert str(selected["gamma"]) == str(selected_config["models"][model_name]["params"]["gamma"])
            summary = pd.DataFrame({
                "설정": ["Baseline", "Selected"],
                "C": [baseline["C"], selected["C"]],
                "gamma": [baseline["gamma"], selected["gamma"]],
                "Track ROC-AUC ↑": [baseline["track_roc_auc"], selected["track_roc_auc"]],
                "Track EER ↓": [baseline["track_eer"], selected["track_eer"]],
                "Track Macro-F1 ↑": [baseline["val_track_macro_f1"], selected["val_track_macro_f1"]],
            })
            display(summary.round(4))

            # 16개 C×gamma 조합을 빠짐없이 같은 색 눈금에 표시한다.
            c_values = sorted(candidates["C"].unique())
            gamma_values = ["scale", "0.001", "0.01", "0.1"]
            grid = candidates.pivot(index="C", columns="gamma", values="track_eer").loc[c_values, gamma_values]
            assert grid.notna().all().all()
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            image = axes[0].imshow(grid.to_numpy(), cmap="YlGnBu", aspect="auto")
            axes[0].set_xticks(range(len(gamma_values)), gamma_values)
            axes[0].set_yticks(range(len(c_values)), [f"{c:g}" for c in c_values])
            axes[0].set_xlabel("gamma")
            axes[0].set_ylabel("C")
            axes[0].set_title("All 16 C × gamma candidates: Track EER ↓")
            for row in range(len(c_values)):
                for col in range(len(gamma_values)):
                    value = grid.iloc[row, col]
                    axes[0].text(col, row, f"{value:.3f}", ha="center", va="center", fontsize=8, color="white" if value > 0.18 else "black")
            from matplotlib.patches import Rectangle
            for item, color in [(baseline, "#3973a6"), (selected, "#ef8b38")]:
                row = c_values.index(item["C"])
                col = gamma_values.index(str(item["gamma"]))
                axes[0].add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, edgecolor=color, linewidth=3))
            fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
        """)
    # The saved CSV calls Macro-F1 val_track_macro_f1; use one common name below.
    return (
        PREAMBLE + model_part + specific
        + f"\nfigure_path = figure_dir / {image_name!r}\n"
        + 'baseline = baseline.rename({"val_track_macro_f1": "track_macro_f1"})\n'
        + 'selected = selected.rename({"val_track_macro_f1": "track_macro_f1"})\n'
        + BAR_PLOT
    )


def cnn_code() -> str:
    return PREAMBLE + dedent("""\
        candidates = pd.read_csv(result_dir / "cnn_tuning.csv").sort_values("candidate_order")
        baseline_info = json.loads((result_dir / "cnn_baseline_selection.json").read_text())
        selected_info = json.loads((result_dir / "cnn_optimized_selection.json").read_text())
        assert len(candidates) == 18 and candidates["status"].eq("complete").all()
        assert candidates["run_id"].nunique() == 1
        assert candidates["run_id"].iloc[0] == baseline_info["run_id"] == selected_info["run_id"]
        baseline = candidates.loc[candidates["trial_id"] == baseline_info["trial"]["trial_id"]].iloc[0]
        selected = candidates.loc[candidates["trial_id"] == selected_info["trial"]["trial_id"]].iloc[0]
        assert int(baseline["candidate_order"]) == baseline_info["trial"]["candidate_order"]
        assert int(selected["candidate_order"]) == selected_info["trial"]["candidate_order"]

        # 선택 설정과 세 가지 Validation Track 지표를 짧은 표로 만든다.
        summary = pd.DataFrame({
            "설정": ["Baseline", "Selected"],
            "learning rate": [baseline["lr"], selected["lr"]],
            "dropout": [baseline["dropout"], selected["dropout"]],
            "batch size": [baseline["batch_size"], selected["batch_size"]],
            "weight decay": [baseline["weight_decay"], selected["weight_decay"]],
            "best epoch": [baseline["best_epoch"], selected["best_epoch"]],
            "Track ROC-AUC ↑": [baseline["track_roc_auc"], selected["track_roc_auc"]],
            "Track EER ↓": [baseline["track_eer"], selected["track_eer"]],
            "Track Macro-F1 ↑": [baseline["track_macro_f1"], selected["track_macro_f1"]],
        })
        display(summary.round(4))

        # 18개 trial의 Validation EER와 선택된 두 trial을 함께 보여 준다.
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(candidates["candidate_order"], candidates["track_eer"], "o", color="#64748b", alpha=0.75)
        axes[0].scatter([baseline["candidate_order"]], [baseline["track_eer"]], s=110, label="Baseline", color="#3973a6", zorder=3)
        axes[0].scatter([selected["candidate_order"]], [selected["track_eer"]], s=110, label="Selected", color="#ef8b38", zorder=3)
        axes[0].set_xlabel("Candidate order (Baseline, Stage 1, Stage 2)")
        axes[0].set_ylabel("Validation Track EER ↓")
        axes[0].set_title("All 18 completed CNN trials")
        axes[0].legend()
        axes[0].grid(alpha=0.2)
        figure_path = figure_dir / "cnn_validation_comparison.png"
    """) + BAR_PLOT


def mert_code() -> str:
    return PREAMBLE + dedent("""\
        config_path = next(result_dir.glob("mert_best_config_mert_binary_*.json"))
        config = json.loads(config_path.read_text())
        candidates = pd.read_csv(result_dir / f"mert_tuning_{config['run_id']}.csv")
        assert len(candidates) == 65 and candidates["run_id"].nunique() == 1
        assert candidates["run_id"].iloc[0] == config["run_id"]
        baseline_config, selected_config = config["baseline"], config["optimized"]

        # layer와 C가 모두 맞는 저장 행을 찾아 Baseline과 선택 후보를 확인한다.
        def find_candidate(spec):
            match = candidates.loc[
                (candidates["layer"] == spec["layer"])
                & np.isclose(candidates["C"], spec["C"])
            ]
            assert len(match) == 1
            return match.iloc[0]

        baseline = find_candidate(baseline_config)
        selected = find_candidate(selected_config)
        summary = pd.DataFrame({
            "설정": ["Baseline", "Selected"],
            "MERT layer": [baseline["layer"], selected["layer"]],
            "LR C": [baseline["C"], selected["C"]],
            "Track ROC-AUC ↑": [baseline["val_track_roc_auc"], selected["val_track_roc_auc"]],
            "Track EER ↓": [baseline["val_track_eer"], selected["val_track_eer"]],
            "Track Macro-F1 ↑": [baseline["val_track_macro_f1"], selected["val_track_macro_f1"]],
        })
        display(summary.round(4))

        # 13개 layer × 5개 C 후보의 EER를 열지도로 보여 준다.
        c_values = sorted(candidates["C"].unique())
        layers = sorted(candidates["layer"].unique())
        grid = candidates.pivot(index="layer", columns="C", values="val_track_eer").loc[layers, c_values]
        assert grid.notna().all().all()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
        image = axes[0].imshow(grid.to_numpy(), cmap="YlGnBu", aspect="auto")
        axes[0].set_xticks(range(len(c_values)), [f"{c:g}" for c in c_values])
        axes[0].set_yticks(range(len(layers)), layers)
        axes[0].set_xlabel("LR C")
        axes[0].set_ylabel("MERT layer (0-based)")
        axes[0].set_title("All 65 layer × C candidates: Track EER ↓")
        for row in range(len(layers)):
            for col in range(len(c_values)):
                value = grid.iloc[row, col]
                axes[0].text(col, row, f"{value:.3f}", ha="center", va="center", fontsize=6, color="white" if value > 0.13 else "black")
        from matplotlib.patches import Rectangle
        for item, color in [(baseline, "#3973a6"), (selected, "#ef8b38")]:
            row = layers.index(int(item["layer"]))
            col = c_values.index(item["C"])
            axes[0].add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, edgecolor=color, linewidth=3))
        fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
        figure_path = figure_dir / "mert_validation_comparison.png"

        # 공통 그래프 코드가 읽을 열 이름을 맞춘다.
        baseline = pd.Series({"track_roc_auc": baseline["val_track_roc_auc"], "track_eer": baseline["val_track_eer"], "track_macro_f1": baseline["val_track_macro_f1"]})
        selected = pd.Series({"track_roc_auc": selected["val_track_roc_auc"], "track_eer": selected["val_track_eer"], "track_macro_f1": selected["val_track_macro_f1"]})
    """) + BAR_PLOT


def execute_new_cells(cells: list) -> list:
    """Run new plotting cells in-process and embed their tables and figures."""
    executed = deepcopy(cells)
    namespace = {"__name__": "__main__"}
    original_display = ipython_display.display
    original_show = plt.show
    for cell in executed:
        if cell.cell_type != "code":
            continue
        outputs = []

        def capture_display(value):
            """Store a displayed table as notebook HTML and plain text."""
            html = value.to_html(index=False) if hasattr(value, "to_html") else repr(value)
            outputs.append(
                nbformat.v4.new_output(
                    "display_data",
                    data={"text/html": html, "text/plain": str(value)},
                    metadata={},
                )
            )

        def capture_show():
            """Store the completed Matplotlib figure in the notebook."""
            figure = plt.gcf()
            buffer = BytesIO()
            figure.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
            outputs.append(
                nbformat.v4.new_output(
                    "display_data",
                    data={"image/png": base64.b64encode(buffer.getvalue()).decode("ascii")},
                    metadata={},
                )
            )
            plt.close(figure)

        stream = StringIO()
        ipython_display.display = capture_display
        plt.show = capture_show
        try:
            with redirect_stdout(stream):
                exec(compile(cell.source, "<new validation comparison cell>", "exec"), namespace)
        finally:
            ipython_display.display = original_display
            plt.show = original_show
        if stream.getvalue():
            outputs.append(nbformat.v4.new_output("stream", name="stdout", text=stream.getvalue()))
        cell.outputs = outputs
    return executed


def append_sections(notebook_name: str, sections: list[tuple[str, str]]) -> None:
    path = ROOT / "archive_notebooks" / notebook_name
    notebook = nbformat.read(path, as_version=4)
    marker_positions = [
        index for index, cell in enumerate(notebook.cells) if MARKER in cell.source
    ]
    trailing_cells = []
    if marker_positions:
        first = marker_positions[0]
        expected_positions = [first + 2 * offset for offset in range(len(sections))]
        if marker_positions != expected_positions:
            raise RuntimeError(f"Unexpected comparison cell layout: {notebook_name}")
        end = first + 2 * len(sections)
        trailing_cells = deepcopy(notebook.cells[end:])
        notebook.cells = notebook.cells[:first]
    original_cells = deepcopy(notebook.cells)
    additions = []
    for title, source in sections:
        additions.append(nbformat.v4.new_markdown_cell(title))
        additions.append(nbformat.v4.new_code_cell(source))
    executed = execute_new_cells(additions)
    assert all(
        cell.cell_type != "code" or cell.outputs for cell in executed
    ), notebook_name
    # Keep execution numbers in reading order without touching earlier cells.
    next_count = max(
        (cell.execution_count or 0 for cell in original_cells if cell.cell_type == "code"),
        default=0,
    )
    for cell in executed:
        if cell.cell_type == "code":
            next_count += 1
            cell.execution_count = next_count
    notebook.cells.extend(executed)
    notebook.cells.extend(trailing_cells)
    nbformat.validate(notebook)
    assert notebook.cells[: len(original_cells)] == original_cells
    nbformat.write(notebook, path)
    print(f"Added {len(sections)} executed comparison(s): {notebook_name}")


def main() -> None:
    append_sections(
        "22_classical_model_tuning.ipynb",
        [
            (
                f"## 8. Logistic Regression — {MARKER}\n\nC 5개의 저장된 결과를 읽어 Baseline과 선택 모델을 비교한다. 모든 값은 Validation 곡 단위이며, Test 결과가 아니다.",
                classical_code("LogisticRegression"),
            ),
            (
                f"## 9. RBF-SVM — {MARKER}\n\nC×gamma 16개 후보의 EER와 Baseline·선택 모델을 함께 본다. 저장된 Validation 결과만 사용한다.",
                classical_code("RBF-SVM"),
            ),
        ],
    )
    append_sections(
        "13B_logmel_cnn_tuning.ipynb",
        [
            (
                f"## 9. CNN — {MARKER}\n\n완료된 18개 trial의 Validation 곡 단위 결과를 읽는다. Baseline과 선택 모델의 학습률·dropout·batch size·weight decay 및 지표를 비교한다.",
                cnn_code(),
            ),
        ],
    )
    append_sections(
        "17B_mert_binary_tuning.ipynb",
        [
            (
                f"## 8. MERT+LR — {MARKER}\n\n13개 layer×5개 C 후보의 저장된 Validation 결과를 읽는다. Baseline과 선택 조합의 곡 단위 지표를 비교한다.",
                mert_code(),
            ),
        ],
    )


if __name__ == "__main__":
    main()
