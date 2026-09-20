"""Build an executed presentation notebook from one completed paired MP3 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat as nbf
import pandas as pd
from nbclient import NotebookClient


def md(value: str):
    return nbf.v4.new_markdown_cell(value)


def code(value: str):
    return nbf.v4.new_code_cell(value)


# 한 모델·조건·수준에 정확히 한 결과만 있는지 확인하고 표에 사용한다.
def metric_row(metrics: pd.DataFrame, model: str, condition: str, level: str = "track"):
    rows = metrics.loc[
        (metrics.model == model)
        & (metrics.condition == condition)
        & (metrics.level == level)
    ]
    if len(rows) != 1:
        raise ValueError(f"Missing {model} {condition} {level} result")
    return rows.iloc[0]


# 입력: 완성된 paired MP3 실행 폴더. 출력: 실제 CSV 수치를 표시한 실행 노트북.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    root = Path(__file__).resolve().parents[1]
    output = args.output or root / "24B_optimized_mp3_robustness.ipynb"
    # 실행된 노트북을 다시 생성해 수치와 출력 셀을 지우지 않도록 막는다.
    if output.exists():
        previous = nbf.read(output, as_version=4)
        if any(
            cell.cell_type == "code" and cell.execution_count is not None
            for cell in previous.cells
        ):
            raise RuntimeError(f"Refusing to overwrite executed notebook: {output}")
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    qc = json.loads(
        (run_dir / "clean_input_reference_qc.json").read_text(encoding="utf-8")
    )
    metrics = pd.read_csv(run_dir / "metrics.csv")
    deltas = pd.read_csv(run_dir / "paired_deltas.csv")
    reference = pd.read_csv(run_dir / "clean_vs_final23.csv")
    # 네 모델×세 조건×두 수준과 두 압축 변화량이 모두 있어야 보고한다.
    if len(metrics) != 24 or len(deltas) != 16 or len(reference) != 8:
        raise ValueError("Paired MP3 run is incomplete")
    models = ("LogisticRegression", "RBF-SVM", "Log-Mel CNN", "Frozen MERT + LR")
    track_rows = []
    for model in models:
        c = metric_row(metrics, model, "clean")
        a = metric_row(metrics, model, "mp3_128")
        b = metric_row(metrics, model, "mp3_64")
        track_rows.append(
            f"| {model} | {c.roc_auc:.4f} | {a.roc_auc:.4f} ({a.roc_auc-c.roc_auc:+.4f}) | "
            f"{b.roc_auc:.4f} ({b.roc_auc-c.roc_auc:+.4f}) | "
            f"{c.eer:.4f} | {a.eer:.4f} | {b.eer:.4f} | "
            f"{c.hter:.4f} | {a.hter:.4f} | {b.hter:.4f} |"
        )
    table = "\n".join(track_rows)
    run_path = run_dir.relative_to(root).as_posix()
    worst_64 = min(
        models,
        key=lambda model: metric_row(metrics, model, "mp3_64").roc_auc
        - metric_row(metrics, model, "clean").roc_auc,
    )
    biggest_real_fpr = max(
        models,
        key=lambda model: metric_row(metrics, model, "mp3_64").real_fpr
        - metric_row(metrics, model, "clean").real_fpr,
    )
    max_ref_auc = (
        reference.loc[reference.level.eq("track"), "delta_roc_auc"].abs().max()
    )

    cells = [
        md(
            f"""# 24B. 새 Optimized 네 모델의 paired MP3 강건성

실행 ID: `{metadata['robustness_run_id']}`. 기존 [공동 최종 Test](23_final_binary_test_comparison.ipynb)의 네 **Optimized** checkpoint와 clean Validation 임계값을 고정했다. 기존 Test의 REAL/FAKE **539 Track·1,572 Segment** 전체를 clean, 128 kbps, 64 kbps로 같은 ID에 대응시켰다. 세 조건 모두 파일 처음부터 전체 디코딩하고 원래 10초 Segment 시작 시각으로 잘랐다. 모델·threshold는 압축 결과로 바꾸지 않았다.

이 계획은 원래 Test 결과가 이미 공개된 뒤 작성됐다. 따라서 새 결과를 역사상 미노출인 사전등록 holdout이라고 주장하지 않는다. 2026-09-13의 기존 MP3 결과는 다른 checkpoint의 과거 실험이다."""
        ),
        md(
            """## 1. 입력·출처 동결 확인

`extract_paired_mp3_features.py`는 세 조건의 266-D 특징과 새 clean Log-Mel을 만들고, 과거 `15`번에서 전체 디코딩으로 생성한 MP3 Log-Mel cache를 실제 압축 파일의 표본과 대조했다. `extract_mert_mp3.py`는 sklearn이 없는 별도 프로세스에서 frozen encoder를 CPU에 올려 세 조건의 13×768 embedding을 추출했다. `score_optimized_mp3_robustness.py`는 저장된 checkpoint와 clean Validation 임계값으로 점수를 계산했다. `run_metadata.json`은 protocol/manifest/report hash, 모델 checkpoint hash와 실행 장비를 보존한다."""
        ),
        code(
            f"""from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image

RUN = Path({run_path!r})
meta = json.loads((RUN / 'run_metadata.json').read_text(encoding='utf-8'))
input_qc = json.loads((RUN / 'clean_input_reference_qc.json').read_text(encoding='utf-8'))
metrics = pd.read_csv(RUN / 'metrics.csv')
deltas = pd.read_csv(RUN / 'paired_deltas.csv')
reference = pd.read_csv(RUN / 'clean_vs_final23.csv')
print('Run:', meta['robustness_run_id'], '| source final:', meta['source_final_run_id'])
print('Protocol SHA-256:', meta['protocol_sha256'])
print('Test:', meta['test_segments'], 'segments /', meta['test_tracks'], 'tracks')
print('Conditions:', meta['conditions'], '| rows:', len(metrics))
display(pd.DataFrame(meta['mert_extraction']).T)
"""
        ),
        md(
            f"""### 결과

세 조건 모두 Test **{metadata['test_segments']} Segment·{metadata['test_tracks']} Track**을 포함했고, 24개 모델×조건×수준 지표가 저장됐다. Clean/MP3는 같은 manifest SHA-256 `{metadata['split_manifest_sha256'][:12]}…`와 사전 고정 protocol SHA-256 `{metadata['protocol_sha256'][:12]}…`을 사용했다. MERT는 CPU에서 Segment 하나씩 추출했고 CNN Test 추론 장비는 `{metadata['cnn_device']}`였다.

### 해석

모델의 학습 상태와 Validation 임계값을 고정한 채 입력의 압축 조건만 바꾸었다. 각 조건의 Track 점수는 같은 Track에 속한 Segment 점수의 평균이다.

### 주의사항

기존 원본 Test 수치가 공개된 이력이 있으며, 세 조건의 decoder·특징 추출과 추론 시간은 장비 부하에 영향을 받는다. 모델 간 원점수 크기는 직접 비교하지 않는다.

### 다음 단계

같은 ID의 clean 대비 128/64 kbps 성능 변화를 본다."""
        ),
        md(
            """## 2. 고정 임계값에서의 세 조건 비교

ROC-AUC와 EER는 점수의 분리력을, HTER·Balanced Accuracy·Macro-F1·REAL FPR·FAKE Miss Rate는 **원래 clean Validation 임계값**으로 얻은 분류를 나타낸다. FAKE AP와 REAL AP를 함께 보며, class 비율은 조건마다 같다. Test EER의 교점은 예측 임계값으로 쓰지 않는다."""
        ),
        code(
            """track = metrics.loc[metrics.level.eq('track')]
display(track[['model','condition','n_real','n_fake','threshold','roc_auc','ap_fake','ap_real',
               'eer','balanced_accuracy','macro_f1','real_fpr','fake_miss_rate','hter']])
display(deltas.loc[deltas.level.eq('track'),
               ['model','condition','delta_roc_auc','delta_ap_real','delta_eer',
                'delta_balanced_accuracy','delta_macro_f1','delta_real_fpr',
                'delta_fake_miss_rate','delta_hter']])
display(Image(filename=str(RUN / 'paired_track_roc_auc.png')))
display(Image(filename=str(RUN / 'paired_track_eer.png')))
"""
        ),
        md(
            f"""### 결과

Track Test는 매 조건 **REAL 45·FAKE 494**로 같았다. 아래 괄호는 새 paired clean 대비 변화량이다.

| 모델 | Clean AUC | 128 kbps AUC (Δ) | 64 kbps AUC (Δ) | Clean EER | 128 EER | 64 EER | Clean HTER | 128 HTER | 64 HTER |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
{table}

### 해석

64 kbps에서 clean 대비 AUC 하락폭이 가장 큰 모델은 **{worst_64}**였다. 압축 후 오류 방향은 모델마다 다르며, 64 kbps REAL FPR 증가폭이 가장 큰 모델은 **{biggest_real_fpr}**였다. AUC·EER와 고정 임계값의 HTER를 함께 읽어 분리력 변화와 분류 오류 변화를 구분한다.

### 주의사항

REAL은 45 Track뿐이어서 한 건의 REAL 오탐 변화가 FPR 약 2.22%p에 해당한다. FAKE 비율이 91.65%이므로 높은 FAKE AP만으로 REAL 탐지 성능을 설명할 수 없다. 모델의 sigmoid·확률·margin 점수는 서로 같은 척도가 아니다.

### 다음 단계

Segment 결과와 같은 ID의 점수 변화, 그리고 새 clean 추출과 23번 결과의 차이를 확인한다."""
        ),
        md(
            """## 3. Segment 변화와 원래 23번 clean 기준 대조

새 paired clean은 MP3와 동일한 전체 디코딩 경로에서 다시 추출했다. 23번의 clean score는 원래 추출 경로에서 만든 값이므로, 섞어서 clean→MP3 변화량을 계산하지 않는다. 입력과 score·지표 차이를 기록해 추출 방식의 영향을 드러낸다."""
        ),
        code(
            """display(metrics.loc[metrics.level.eq('segment'),
                ['model','condition','n_real','n_fake','roc_auc','ap_fake','ap_real','eer',
                 'balanced_accuracy','macro_f1','real_fpr','fake_miss_rate','hter']])
display(reference[['model','level','segment_score_mean_abs_diff','segment_score_max_abs_diff',
                   'delta_roc_auc','delta_eer','delta_balanced_accuracy','delta_hter']])
print('Clean input comparison with final 23:', json.dumps(input_qc,ensure_ascii=False,indent=2))
"""
        ),
        md(
            f"""### 결과

새 paired clean과 23번 clean의 입력 차이는 handcrafted 평균 절댓값 **{qc['handcrafted']['mean_abs_diff']:.6g}**, Log-Mel 평균 절댓값 **{qc['logmel']['mean_abs_diff']:.6g}**, MERT 선택 layer 평균 절댓값 **{qc['mert_selected_layer']['mean_abs_diff']:.6g}**였다. 두 clean 추출의 Track AUC 차이의 최대 절댓값은 네 모델 중 **{max_ref_auc:.6f}**였다. 각 모델·수준의 정확한 차이와 Segment 조건별 전 지표는 위 표 및 CSV에 있다.

### 해석

23번의 clean과 새 paired clean을 동일한 전처리 결과로 취급할 수 있는지 직접 확인했다. 강건성 Delta는 일관된 전체 디코딩 경로의 새 paired clean에서 계산했다.

### 주의사항

입력 차이는 압축 자체 외에 전체 디코딩과 이전 segment 단위 로딩 방식의 차이를 포함할 수 있다. 이 비교는 Test 기반 재튜닝이 아니며, 저장된 clean Validation 임계값은 그대로 유지했다.

### 다음 단계

모델별 압축 민감도와 현재 평가의 한계를 정리한다."""
        ),
        md(
            """## 4. 결론과 한계

### 결과

네 Optimized 모델의 clean/128/64 kbps Segment·Track 지표, 동일 ID의 raw score와 변화량은 `results/model_robustness/mp3/` 아래 이 실행 ID 디렉터리에 저장했다. 계산량은 `inference_times.csv`, MERT 추출 시간은 `run_metadata.json`에 있다.

### 해석

이 표는 고정 모델이 압축 입력에 얼마나 민감한지 비교한다. 분리력과 REAL 오탐·FAKE 누락을 함께 보고, 한 지표만으로 모델의 전체 강건성을 단정하지 않는다.

### 주의사항

이 실험은 이미 공개된 Test 결과를 본 뒤 설계한 후속 분석이다. 128/64 kbps와 한 가지 transcode 경로만 다루며 다른 codec·플랫폼 처리를 대표하지 않는다. FMA REAL과 Echoes TTA FAKE의 출처·장르·제작 방식 차이, REAL Track 45개의 작은 표본, MERT의 외부 사전학습 데이터 중복 불확실성을 남긴다. Unseen-generator 일반화는 [별도 고정 설정 재학습 실험](docs/UNSEEN_FIXED_PROTOCOL.md)으로 평가한다.

### 다음 단계

독립 원곡 단위의 불확실성 평가와 다른 codec을 포함한 외부 검증으로 결과 범위를 확인한다."""
        ),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata.language_info = {"name": "python"}
    # 저장된 결과 CSV를 읽는 표시 셀까지 실제 커널에서 실행한다.
    NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(root)}},
    ).execute()
    nbf.write(notebook, output)
    print(output)


if __name__ == "__main__":
    main()
