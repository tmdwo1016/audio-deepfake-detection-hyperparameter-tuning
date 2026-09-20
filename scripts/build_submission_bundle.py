"""Build the experiment-record submission ZIP without local audio/cache/checkpoints.

Run from any directory: ``python scripts/build_submission_bundle.py``.
The ZIP contains the report, 32 executed lab notebooks, source, compact results,
and metadata manifests.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "AI_music_project_submission_20260920.zip"
# 검토에 필요한 작은 파일만 모으고, 수 GB 오디오·cache·checkpoint는 제외한다.
INCLUDE_DIRS = (
    "archive_notebooks", "docs", "scripts", "src", "tests",
    "results", "data/metadata",
)
EXCLUDE_PARTS = {"__pycache__", ".ipynb_checkpoints"}


def submission_files() -> list[Path]:
    """Select the archive notebooks, documentation, and compact results."""
    # README와 변경 기록은 루트에, 실행 기록은 archive_notebooks/에 둔다.
    files = [ROOT / name for name in ("README.md", "CHANGELOG.md", ".gitignore")]
    # 결과 그림·CSV와 코드·안내 문서를 재귀적으로 찾는다.
    for folder in INCLUDE_DIRS:
        files.extend((ROOT / folder).rglob("*"))
    # 중복, 생성된 ZIP 자신, 편집기와 Python cache 파일을 제거한다.
    return sorted(
        {
            path
            for path in files
            if path.is_file()
            and path != OUTPUT
            and path.name != ".DS_Store"
            and not EXCLUDE_PARTS.intersection(path.parts)
        }
    )


def main() -> None:
    """Write the ZIP and verify its entries can be decompressed."""
    files = submission_files()
    # 파일마다 크기와 SHA-256을 기록해 ZIP에서 내용이 바뀌었는지 확인할 수 있게 한다.
    manifest = {
        "package": OUTPUT.stem,
        "purpose": "32개 실행 노트북·문서·코드·작은 결과의 검토용 묶음",
        "excludes": [
            "data/raw",
            "data/processed",
            "checkpoints",
            "과거 v3 제출 패키지",
        ],
        "rerun_note": (
            "원본 audio, processed cache, checkpoint가 없으면 전체 재실행은 불가능하다. "
            "docs/SUBMISSION_GUIDE.md 참고."
        ),
        "files": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
            for path in files
        ],
    }
    # 상대 경로를 유지해야 ZIP을 풀었을 때 README의 링크가 연결된다.
    with ZipFile(OUTPUT, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr(
            "SUBMISSION_MANIFEST.json",
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )
        for path in files:
            archive.write(path, path.relative_to(ROOT))

    # 압축 손상과 누락 항목을 먼저 확인하고 제출 파일 경로를 출력한다.
    with ZipFile(OUTPUT) as archive:
        bad_entry = archive.testzip()
        if bad_entry is not None:
            raise RuntimeError(f"Corrupt ZIP entry: {bad_entry}")
        if len(archive.namelist()) != len(files) + 1:
            raise RuntimeError("Submission ZIP entry count differs from manifest")
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    print(
        f"{OUTPUT.name}: {len(files)} files, {OUTPUT.stat().st_size} bytes, SHA-256 {digest}"
    )


if __name__ == "__main__":
    main()
