"""Generate the final binary comparison notebook without erasing execution results."""

import argparse
from pathlib import Path
import nbformat as nbf


def md(value):
    return nbf.v4.new_markdown_cell(value)


def code(value):
    return nbf.v4.new_code_cell(value)


# 노트북 셀은 선택 산출물 검사 → 고정 Test → 추론 → 해석 순서로 배치한다.
cells = [
    md(
        """# 23. 네 모델 Baseline·Optimized 최종 Test 비교

22번 LR/SVM, 13B CNN, 17B MERT의 Train/Validation 탐색을 모두 마친 뒤 실행한다. 기존 09/13/17의 Test 출력은 **과거 실험**이다. 네 모델의 Baseline과 Optimized를 이 단계에서 같은 Test 집합으로 평가한다. 선택된 Train 학습 모델을 그대로 사용하고 Train+Validation 재학습과 Test 기반 설정 변경은 하지 않는다.

REAL=0, FAKE=1이며 높은 score는 FAKE를 뜻한다. LR/MERT는 FAKE 열 확률, SVM은 margin, CNN은 sigmoid(logit)을 사용한다. 모델 간 score 크기는 직접 비교하지 않는다. Track score는 동일 Track의 Segment score 평균이다."""
    ),
    md(
        """## 1. 선택 산출물 동결 검사

모든 checkpoint, Validation의 Segment/Track 임계값, run ID, manifest hash, CNN cache index와 MERT revision을 확인한 뒤 Test를 연다. 공통 지표는 ROC-AUC, FAKE/REAL AP, 보간 EER, Balanced Accuracy, Macro-F1, REAL FPR, FAKE Miss Rate, confusion matrix와 HTER다. AP는 average_precision_score이며 사다리꼴 PR-AUC와 다르다. Test HTER는 저장된 Validation 임계값의 오류율 평균이고 Test EER은 분리력 지표다."""
    ),
    code(
        """from pathlib import Path
import pandas as pd
from IPython.display import display
from src.modeling_final_run import (
    preflight, load_fixed_test, infer_classical, infer_cnn, infer_mert, finish_test
)

# Test를 읽기 전에 네 모델의 Baseline/Optimized 선택을 모두 검사한다.
context = preflight(Path.cwd())
print("Final run:", context["final_run_id"])
print("Component runs:", context["run_ids"])
print("Manifest hash:", context["manifest_hash"])
"""
    ),
    md(
        """### 결과

미실행. 위 출력의 run ID와 manifest hash를 기록한다.

### 해석

모델 선택 산출물과 입력 manifest의 출처가 일치하는지 확인한다.

### 주의사항

과거 Test 결과가 이미 공개되었으므로 이번 자료를 역사상 완전히 미사용인 Test라고 주장하지 않는다.

### 다음 단계

고정 Test의 ID와 class 구성을 확인한다."""
    ),
    md(
        """## 2. 고정 Test 입력

기존 original_audio group split을 다시 만들지 않는다. REAL/FAKE mapping, group 중복, Track label과 266-D feature 순서를 검증한다. 모델별 실패 sample을 조용히 제외하지 않는다."""
    ),
    code(
        """# 모든 Validation 선택이 동결된 뒤 처음으로 Test를 연다.
test_meta, X_test, manifest = load_fixed_test(context)
class_counts = (manifest.groupby(["split", "label"], sort=False)
                .agg(original_audio=("original_audio", "nunique"),
                     tracks=("track_sample_id", "nunique"),
                     segments=("segment_id", "nunique")).reset_index())
display(class_counts)
print("Test segments:", len(test_meta), "Tracks:", test_meta.track_sample_id.nunique())
print("Feature columns:", len(X_test.columns))
"""
    ),
    md(
        """### 결과

미실행. 실제 split별 original_audio, Track, Segment class 수를 위 표에서 기록한다.

### 해석

같은 Test 집합과 class mapping을 네 모델의 Baseline/Optimized에 적용한다.

### 주의사항

FAKE가 다수이므로 높은 FAKE AP 하나만으로 불균형 영향이 해결되었다고 해석하지 않는다.

### 다음 단계

동결한 Train 모델로 Test score만 계산한다."""
    ),
    md(
        """## 3. LR와 RBF-SVM Test 추론

Pipeline의 StandardScaler는 Train에서만 fit되었다. LR은 classes_의 FAKE 열 predict_proba, SVM은 probability=False의 decision_function margin을 사용한다. 두 모델에 class_weight=balanced가 적용되었다. 동일 후보의 Baseline과 Optimized는 score를 재사용한다. C는 정규화 강도와 관련되고 SVM gamma는 RBF 곡면의 영향을 조절한다. 이 모델들은 CNN처럼 직접 learning rate, batch size, epoch를 지정하지 않는다."""
    ),
    code(
        """# Train-only scaler와 분류기가 함께 저장된 Pipeline으로 추론한다.
classical_scores, classical_times = infer_classical(context, test_meta, X_test)
display(pd.DataFrame(classical_times))
"""
    ),
    md(
        """### 결과

미실행. LR와 RBF-SVM의 Baseline/Optimized Test 추론 시간과 score가 모두 저장되었는지 확인해 기록한다. 성능 지표는 네 모델의 score가 모두 준비된 뒤 6절에서 함께 계산한다.

### 해석

같은 후보가 Baseline과 Optimized에 모두 선택되었다면 동일 checkpoint의 예측을 재사용했는지 확인한다.

### 주의사항

SVM margin은 확률이 아니며, 여기서 계산한 Test score로 후보나 임계값을 다시 고르지 않는다.

### 다음 단계

동일한 Test Segment를 CNN의 동결 checkpoint로 추론한다."""
    ),
    md(
        """## 4. Log-Mel CNN Test 추론

기존 네 convolution block과 Log-Mel 설정을 유지한다. 입력 [1,128,T]의 1은 채널, 128은 Mel bin, T는 시간 frame이다. 기존 24 kHz, n_fft=1024, hop=240, center=True에서는 T=1001이다. 1001은 10초 길이만으로 정해지지 않는다. 128 Mel bin은 128 handcrafted feature가 아니다. CNN sigmoid score를 보정된 실제 확률로 단정하지 않는다."""
    ),
    code(
        """# Validation Track EER로 선택된 best epoch checkpoint를 복원한다.
cnn_scores, cnn_times, cnn_device = infer_cnn(context, test_meta, manifest)
print("CNN device:", cnn_device)
display(pd.DataFrame(cnn_times))
"""
    ),
    md(
        """### 결과

미실행. CNN Baseline/Optimized 추론 시간, 장비와 예측 Segment 수를 기록한다. 실제 ROC-AUC와 EER은 6절의 공동 Test 평가 결과를 옮겨 적는다.

### 해석

두 CNN checkpoint의 차이와 선택된 Validation best epoch를 저장된 선택 정보로 설명한다.

### 주의사항

가중 BCE로 학습한 sigmoid score를 보정된 실제 확률로 해석하지 않는다. Test 결과로 checkpoint를 변경하지 않는다.

### 다음 단계

동결된 MERT encoder로 같은 Test Segment의 embedding을 추출한다."""
    ),
    md(
        """## 5. Frozen MERT Test embedding과 LR 추론

MERT-v1-95M의 24 kHz processor를 사용한다. hidden_states[0]은 첫 Transformer block 이전, 1–12는 각 block 출력으로 총 13 representation level이다. 각 level은 768-D이며 13×768을 붙여 사용하지 않는다. frozen encoder를 eval/inference mode로 두고 padding 없는 유효 frame 평균을 계산한다. Test cache는 모든 선택이 끝난 이 단계에서 만든다. MERT는 외부 사전학습을 사용한다."""
    ),
    code(
        """# 동일 Test segment의 13개 layer를 한 번 추출하고 선택 layer만 사용한다.
mert_scores, mert_times, mert_details = infer_mert(context, test_meta)
print("MERT extraction:", mert_details)
display(pd.DataFrame(mert_times))
"""
    ),
    md(
        """### 결과

미실행. MERT 모델 로드·Test embedding 추출·LR 추론의 실제 시간과 선택 layer/C를 기록한다. 성능 지표는 6절의 공동 Test 평가에서 기록한다.

### 해석

Baseline과 Optimized가 같은 layer/C를 사용하면 동일한 Train 학습 head의 score를 재사용했는지 확인한다.

### 주의사항

MERT는 외부 사전학습 표현을 사용한다. 모델 간 계산 시간을 비교할 때 embedding 추출과 head 추론 시간을 구분한다.

### 다음 단계

저장된 Validation 임계값으로 여덟 변형의 Segment/Track Test 지표를 함께 계산한다."""
    ),
    md(
        """## 6. 공통 Test 평가와 Baseline 비교

Segment와 Track에 저장된 서로 다른 Validation 임계값을 적용한다. prediction은 score >= threshold다. Test EER을 계산하지만 그 threshold로 Test 분류를 바꾸지 않는다. Test Balanced Accuracy, Macro-F1, REAL FPR, FAKE Miss Rate, confusion matrix 및 HTER에는 Validation 임계값을 사용한다. 네 모델의 Baseline과 Optimized 결과를 모두 보존한다."""
    ),
    code(
        """# 정확히 같은 Test ID의 여덟 score를 한 최종 단계에서 평가한다.
all_scores = {**classical_scores, **cnn_scores, **mert_scores}
all_times = classical_times + cnn_times + mert_times
metrics, track_comparison, segment_comparison, optimized = finish_test(
    context, test_meta, all_scores, all_times, cnn_device, mert_details
)
import json

# 선택 정보와 Test 지표를 한 표에 함께 표시한다.
best_hyperparameters = {
    "LogisticRegression": {"C": context["classical_best"]["models"]["LogisticRegression"]["params"]["C"]},
    "RBF-SVM": {key: context["classical_best"]["models"]["RBF-SVM"]["params"][key]
                for key in ("C", "gamma")},
    "Log-Mel CNN": {key: context["cnn_selection"]["optimized"]["trial"][key]
                    for key in ("lr", "dropout", "weight_decay", "batch_size", "max_epochs", "patience")},
    "Frozen MERT + LR": {key: context["mert_checkpoint"]["optimized"][key]
                         for key in ("layer", "C")},
}
track_display = track_comparison.copy()
track_display["best_hyperparameters"] = track_display["model"].map(
    lambda model: json.dumps(best_hyperparameters[model], ensure_ascii=False)
)
display(track_display[[
    "model", "baseline_roc_auc", "optimized_roc_auc", "delta_roc_auc",
    "baseline_eer", "optimized_eer", "delta_eer",
    "baseline_balanced_accuracy", "optimized_balanced_accuracy",
    "baseline_macro_f1", "optimized_macro_f1", "best_hyperparameters"
]])
display(optimized.loc[optimized.level.eq("track"), [
    "model", "n_real", "n_fake", "real_prevalence", "fake_prevalence",
    "roc_auc", "ap_fake", "ap_real", "eer",
    "balanced_accuracy", "macro_f1", "real_fpr", "fake_miss_rate", "hter"
]])
print("CSV, raw scores and PNG saved to", context["out"])
"""
    ),
    md(
        """### 결과

미실행. 실제 Track 표에서 Baseline/Optimized 변화와 네 Optimized 모델의 ROC-AUC, 두 AP, EER, Balanced Accuracy, Macro-F1, REAL FPR, FAKE Miss Rate, HTER를 기록한다. Segment 표는 별도 CSV에서 기록한다. Delta는 Optimized−Baseline이다.

### 해석

AUC 증가는 개선, EER 감소는 개선 방향이다. 강한 모델은 분리력뿐 아니라 REAL 오탐, FAKE 누락, 계산량과 강건성을 함께 고려한다. Segment와 Track 차이는 score 집계 및 평가 단위의 차이로 설명한다.

### 주의사항

FMA REAL과 Echoes TTA FAKE의 출처·장르·코덱·제작 방식 차이가 score에 반영될 수 있다. MERT 사전학습 데이터와 평가 데이터의 중복 여부를 확인하지 못했다면 한계로 적는다.

### 다음 단계

실제 수치로 결론을 작성하고 held-out-generator와 paired MP3 평가로 이어간다."""
    ),
    md(
        """## 7. 실행 후 결론

실행 전에는 성능 숫자를 채우지 않는다. 실제 결과로 다음에 답한다.

1. 초기 설정에서 가장 강한 모델은?
2. 모델별 최적화 전후 ROC-AUC, EER, 오탐, 누락 변화는?
3. 이 search space와 Validation 기준에서 선택된 모델 중 강한 모델은?
4. Baseline과 Optimized의 순위가 바뀌었는가?
5. Train 불균형을 각 모델에서 어떻게 처리했는가?
6. 이번 선택 과정에서 Test 지표를 사용하지 않았음을 무엇으로 확인했는가?
7. Segment와 Track 결과는 어떻게 달랐는가?
8. Unseen generator와 MP3에서 무엇을 검증해야 하는가?

**결과 / 해석 / 주의사항 / 다음 단계**를 실제 CSV와 그림의 수치로 채운다. best는 전체 parameter 공간의 절대 최적값이 아니다. Unseen generator는 대상 생성기를 Train/Validation과 선택 절차에서 제외한 재학습이 필요하다. MP3는 동일 Test REAL/FAKE track의 paired 변환과 clean Validation 임계값을 사용한다. 기존 attribution 및 탐색적 분석은 별도 과거 run으로 구분한다."""
    ),
]

# 생성한 셀을 새 노트북으로 묶고 실행 가능한 커널 정보를 넣는다.
notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata.kernelspec = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
notebook.metadata.language_info = {"name": "python"}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="Notebook path; defaults to project 23_final_binary_test_comparison.ipynb",
)
args = parser.parse_args()
target = (
    args.output
    or Path(__file__).resolve().parents[1] / "23_final_binary_test_comparison.ipynb"
)
# 이미 실행된 노트북의 실제 결과를 템플릿 재생성으로 덮어쓰지 않는다.
if target.exists():
    previous = nbf.read(target, as_version=4)
    if any(
        cell.cell_type == "code" and (cell.execution_count is not None or cell.outputs)
        for cell in previous.cells
    ):
        raise RuntimeError(f"Refusing to overwrite executed notebook: {target}")
target.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, target)
print(target)
