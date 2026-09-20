"""Execute notebook 23 in a real kernel, saving rich outputs after each cell.

The notebook itself calls preflight before opening the fixed Test split.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "23_final_binary_test_comparison.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    # 이미 실행된 최종 Test 노트북의 결과를 다시 덮어쓰지 않는다.
    if any(
        cell.cell_type == "code" and cell.execution_count is not None
        for cell in notebook.cells
    ):
        raise RuntimeError(
            "Final notebook has executed cells; inspect before another Test run"
        )
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("NUMBA_CACHE_DIR", tempfile.gettempdir())
    client = NotebookClient(
        notebook, timeout=None, kernel_name="python3", allow_errors=False
    )
    client.reset_execution_trackers()
    # 셀 하나가 끝날 때마다 노트북을 저장해 중간 결과도 검토할 수 있다.
    with client.setup_kernel(cwd=str(root)):
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            print(f"Running final notebook cell {index}", flush=True)
            try:
                client.execute_cell(cell, index)
            finally:
                # A completed cell is reviewable even if a later inference fails.
                temporary = path.with_suffix(".ipynb.tmp")
                nbformat.write(notebook, temporary)
                os.replace(temporary, path)
    print("Final Test notebook execution complete:", path, flush=True)


if __name__ == "__main__":
    main()
