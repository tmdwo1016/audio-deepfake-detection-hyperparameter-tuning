"""Execute frozen fixed-transfer CNN Test and add measured Korean Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat
import pandas as pd
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "27_cnn_unseen_fixed_transfer.ipynb"


def f(value: float) -> str:
    return f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    out = ROOT / "results/cnn_unseen_fixed_transfer" / args.run_id
    freeze = out / "selections_frozen.json"
    # 두 생성기의 Validation 선택이 동결되지 않았다면 Test 노트북을 실행하지 않는다.
    if not freeze.exists():
        raise RuntimeError("Both fresh fixed-transfer CNN fits must freeze before Test")
    classical = list(
        (ROOT / "results/unseen_fixed_transfer_classical_mert").glob(
            "*/primary_test_metrics.csv"
        )
    )
    if not classical:
        raise RuntimeError("The other three fixed-transfer models have not completed")
    nb = nbformat.read(NOTEBOOK, as_version=4)
    if args.run_id not in nb.cells[2].source:
        raise ValueError("Notebook run ID differs from frozen run")
    print("Executing fixed-transfer CNN notebook and frozen Test", flush=True)
    # 노트북의 Test 셀을 실행한 뒤 저장된 CSV에서 실제 수치를 읽는다.
    nb = NotebookClient(
        nb,
        timeout=None,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()
    nbformat.write(nb, NOTEBOOK)

    config = json.loads((out / "run_config.json").read_text())
    counts = pd.read_csv(out / "filtered_train_val_counts.csv")
    chosen = pd.read_csv(out / "selection_summary.csv")
    # 결과 설명에는 예시 숫자가 아니라 실행 산출물의 지표를 넣는다.
    primary = pd.read_csv(out / "primary_test_metrics.csv")
    paired = pd.read_csv(out / "paired_source_test_metrics.csv")
    train = counts[counts.split == "train"].set_index("holdout_generator")
    val = counts[counts.split == "val"].set_index("holdout_generator")

    nb.cells[
        3
    ].source = f"""### 결과

실행 ID는 `{args.run_id}`, 장치는 `{config['device']}`였다. 기존 Log-Mel cache shape는 **(10077, 128, 1001)**이고, Main CNN checkpoint SHA-256은 `{config['main_checkpoint_sha256']}`, 새 프로토콜 SHA-256은 `{config['protocol_sha256']}`였다. 고정 설정은 LR `3e-4`, dropout `0.3`, batch `16`, weight decay `1e-3`, AdamW, max epoch `30`, patience `4`로 검증됐다.

### 해석

13B의 선택 설정과 manifest/cache 표현을 그대로 가져왔다. Main checkpoint는 **설정 검증에만 읽었고 가중치를 새 모델로 복사하지 않았다.** Log-Mel의 1001 time frame은 기존 24 kHz·hop 240·center STFT 설정에서 나온 실제 cache shape다.

### 주의사항

Main 설정은 MusicGen/Udio가 포함된 Validation에서 선택됐다. 따라서 대상 생성기를 새 학습에서 제외했어도 하이퍼파라미터 선택까지 미노출된 실험은 아니다. hash 일치는 출처 차이의 영향을 제거하지 않는다.

### 다음 단계

대상별 Train·Validation에서 FAKE 제거와 class 수를 확인한다."""

    lines = []
    for holdout in ("musicgen", "udio"):
        tr, va = train.loc[holdout], val.loc[holdout]
        lines.append(
            f"- **{holdout}**: Train {int(tr.n_segments):,} Segment "
            f"(REAL {int(tr.n_real_segments):,}, FAKE {int(tr.n_fake_segments):,}), "
            f"Validation {int(va.n_segments):,} Segment "
            f"(REAL {int(va.n_real_segments):,}, FAKE {int(va.n_fake_segments):,}); "
            f"target FAKE 행 **{int(tr.target_fake_rows)}/{int(va.target_fake_rows)}**."
        )
    nb.cells[6].source = (
        """### 결과

"""
        + "\n".join(lines)
        + """

### 해석

대상 FAKE가 두 split에서 모두 제거됐고 다른 데이터는 기존 group split에 남았다. 각 모델의 pos_weight는 대응 Train segment의 REAL/FAKE 비율에서 새로 계산됐다.

### 주의사항

FAKE가 다수 class이고 source/genre/codec 차이가 남는다. class weighting은 track 길이별 기여도 차이를 모두 제거하지 않는다.

### 다음 단계

같은 고정 하이퍼파라미터로 MusicGen/Udio 모델을 각각 처음부터 학습한다."""
    )

    lines = []
    for row in chosen.itertuples():
        lines.append(
            f"- **{row.holdout_generator}**: {int(row.epochs_run)} epoch 실행, "
            f"best epoch **{int(row.best_epoch)}**, {int(row.updates):,} optimizer step, "
            f"학습 {row.elapsed_sec/60:.1f}분. Train REAL/FAKE "
            f"{int(row.train_real)}/{int(row.train_fake)}, pos_weight **{row.pos_weight:.6f}**. "
            f"Validation Segment AUC/EER {f(row.val_segment_auc)}/{f(row.val_segment_eer)}, "
            f"Track AUC/EER **{f(row.val_track_auc)}/{f(row.val_track_eer)}**; "
            f"Segment/Track 임계값 **{row.segment_threshold:.6f}/{row.track_threshold:.6f}**."
        )
    nb.cells[9].source = (
        """### 결과

"""
        + "\n".join(lines)
        + """

### 해석

두 모델 모두 Main의 LR·dropout·batch·weight decay를 고정한 fresh fit이다. 각자의 필터된 Validation에서만 EER/AUC 기준 최고 epoch와 별도 Segment·Track threshold를 선택했다.

### 주의사항

Validation 최고 epoch 성능은 일반화 성능이 아니다. Early stopping 때문에 두 대상의 epoch 수·계산량이 다를 수 있다. seed 42가 장치별 bitwise 재현을 보장하지 않는다.

### 다음 단계

두 checkpoint와 threshold를 동결한 상태로 원래 Test의 사전 정의 primary cohort를 평가한다."""
    )

    track = primary[primary.level == "track"].set_index("holdout_generator")
    segment = primary[primary.level == "segment"].set_index("holdout_generator")
    lines = []
    for holdout in ("musicgen", "udio"):
        t, s = track.loc[holdout], segment.loc[holdout]
        lines.append(
            f"- **{holdout}**: Track {int(t.n)}개(REAL {int(t.n_real)}, FAKE {int(t.n_fake)}), "
            f"Segment {int(s.n)}개. Track AUC/EER **{f(t.roc_auc)}/{f(t.eer)}**, "
            f"FAKE/REAL AP **{f(t.ap_fake)}/{f(t.ap_real)}**, "
            f"Balanced Accuracy {f(t.balanced_accuracy)}, Macro-F1 {f(t.macro_f1)}, "
            f"REAL FPR {f(t.real_fpr)}, FAKE miss {f(t.fake_miss_rate)}, "
            f"HTER **{f(t.hter)}**. Segment AUC/EER {f(s.roc_auc)}/{f(s.eer)}."
        )
    nb.cells[12].source = (
        """### 결과

"""
        + "\n".join(lines)
        + """

### 해석

AUC/EER는 대상 Test의 점수 분리력이고, Balanced Accuracy·Macro-F1·FPR·miss·HTER는 **해당 대상 Validation 임계값**을 그대로 적용한 분류 결과다. Track 점수는 같은 track Segment score의 평균이다. FAKE AP와 REAL AP를 각 class 비율과 함께 본다.

### 주의사항

23번 전체 Test와 대상별 subset은 표본·class 비율이 달라 직접 수치 비교하지 않는다. Main 하이퍼파라미터 선택에 대상 생성기 Validation이 포함됐고 과거 Test 결과도 공개됐다. REAL은 45 Track으로 작으며 FMA/Echoes TTA 출처 차이가 남는다. Udio REAL 21 원곡 group에는 해당 FAKE 짝이 없다.

### 다음 단계

Udio의 원곡 대응 24-group subset을 보조 분석으로 확인한다."""
    )

    pair = paired[paired.level == "track"].iloc[0]
    udio = track.loc["udio"]
    nb.cells[
        15
    ].source = f"""### 결과

Udio 원곡 대응 subset의 Track은 **{int(pair.n)}개**(REAL {int(pair.n_real)}, FAKE {int(pair.n_fake)})이고 AUC/EER **{f(pair.roc_auc)}/{f(pair.eer)}**, HTER **{f(pair.hter)}**였다. 모든 REAL 45개를 사용한 primary Udio Track {int(udio.n)}개에서는 AUC/EER **{f(udio.roc_auc)}/{f(udio.eer)}**, HTER **{f(udio.hter)}**였다.

### 해석

두 값은 REAL 원곡 coverage가 다른 cohort에서 계산됐다. 보조 subset은 이미 저장된 primary score를 필터했고 동일 checkpoint와 Validation 임계값을 사용했다.

### 주의사항

subset은 FAKE가 있는 원곡으로 조건화되고 class 비율도 다르다. 더 좋은 쪽을 선택 결과로 삼지 않는다.

### 다음 단계

epoch 기록과 primary Test 지표를 그림으로 확인한다."""

    nb.cells[
        18
    ].source = f"""### 결과

두 대상의 Train loss·Validation Track EER epoch 곡선과 primary Test Track EER 그림을 저장했다. 선택 epoch는 MusicGen **{int(chosen.set_index('holdout_generator').loc['musicgen','best_epoch'])}**, Udio **{int(chosen.set_index('holdout_generator').loc['udio','best_epoch'])}**이며, Test Track EER는 각각 **{f(track.loc['musicgen','eer'])}**, **{f(track.loc['udio','eer'])}**였다.

![고정 설정 CNN 학습 곡선과 대상별 Test Track EER](results/cnn_unseen_fixed_transfer/{args.run_id}/cnn_unseen_fixed_training_test.png)

### 해석

Train 손실은 최적화 진행을, Validation EER는 checkpoint 선택을, Test EER는 동결된 모델의 대상 cohort 분리력을 보여준다.

### 주의사항

두 대상의 Test FAKE와 원곡 coverage가 달라 차이를 생성기 자체의 난도로 단정할 수 없다. 그림에 원곡 단위 신뢰구간은 없다.

### 다음 단계

동일 primary Test ID의 LR·SVM·MERT 고정 설정 전이 결과와 CNN을 공통표에서 비교한다."""

    nbformat.validate(nb)
    nbformat.write(nb, NOTEBOOK)
    print("Executed and annotated", NOTEBOOK, flush=True)


if __name__ == "__main__":
    main()
