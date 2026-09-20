"""Run fresh fixed-hyperparameter CNN fits, then a separate frozen Test phase."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# 파일 경로로 직접 실행해도 src를 찾을 수 있도록 루트를 등록한다.
sys.path.insert(0, str(ROOT))

from src.cnn_unseen_fixed import FixedTransferCNNExperiment, HOLDOUTS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--phase", choices=("fit", "test"), required=True)
    args = parser.parse_args()
    experiment = FixedTransferCNNExperiment(ROOT, args.run_id)
    # fit 단계에서는 두 holdout의 선택만 끝내고 Test는 열지 않는다.
    if args.phase == "fit":
        for holdout in HOLDOUTS:
            experiment.fit_holdout(holdout)
        # 두 선택이 모두 끝났다는 동결 파일을 만든 뒤 Test phase로 넘긴다.
        experiment.freeze_both()
        print("BOTH_FIXED_TRANSFER_SELECTIONS_FROZEN", flush=True)
    else:
        print(experiment.final_test().to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
