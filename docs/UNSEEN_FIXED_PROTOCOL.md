# Unseen generator: 고정 하이퍼파라미터 전이 실험

2026-09-20 사용자 지시에 따라 **새 primary protocol**로 고정했다. 이전 [강건성 프로토콜](ROBUSTNESS_PROTOCOL.md)의 holdout별 36개 CNN 후보 탐색은 중단하며, 이미 완료된 trial과 Classical/MERT의 holdout별 탐색은 supplementary 산출물로 보존한다. 이 문서는 새로운 Test score 계산 전에 작성했다.

## 목적과 설정

Main binary experiment에서 Validation으로 선택한 설정을 생성기별로 다시 고르지 않고 그대로 가져와, 대상 생성기를 제외한 Train에서 모델 가중치만 처음부터 다시 학습한다. 원래 선택 과정에는 MusicGen/Udio가 포함되어 있었으므로 이는 **완전히 미노출인 하이퍼파라미터 선택 실험이 아니라 고정 설정 전이 실험**이다. 기존 Test와 대상 생성기의 과거 분석 결과도 이미 공개됐다.

- 대상: `musicgen`, `udio` 각각 별도 실행.
- 기존 `original_audio` group split, 10초 Segment 경계, cache와 class mapping(REAL=0, FAKE=1)을 그대로 사용한다.
- 대상 FAKE를 원래 Train·Validation에서 완전히 제외하고 REAL과 다른 FAKE는 원래 split에 둔다. 대상 행을 Test로 옮기지 않는다.
- Main best 설정은 [22번](../22_classical_model_tuning.ipynb), [13B](../13B_logmel_cnn_tuning.ipynb), [17B](../17B_mert_binary_tuning.ipynb)의 저장된 선택 JSON·checkpoint에서 읽고 hash로 확인한다. LR `C=0.01`; RBF-SVM `C=10, gamma=0.001`; Frozen MERT+LR `layer=4, C=0.01`; CNN `lr=3e-4, dropout=0.3, batch_size=16, weight_decay=1e-3`, 기존 4 ConvBlock 구조와 AdamW를 사용한다.
- 네 모델은 대상별 필터된 Train에 **처음부터** 새로 적합한다. LR/SVM/MERT는 Train-only StandardScaler와 balanced class weight를 사용한다. CNN은 매 대상의 실제 Train REAL/FAKE Segment 수로 `pos_weight=train_real/train_fake`를 다시 계산한다. CNN은 seed=42로 새 가중치를 초기화하고 max_epochs=30, patience=4를 사용한다.
- 하이퍼파라미터는 고정한다. CNN best epoch/checkpoint는 대상별 필터된 Validation Track EER 최소 → AUC 최대 → 이른 epoch 순서로 선택한다. 모든 모델의 Segment·Track 임계값은 각각 해당 필터된 Validation에서만 선택한다.
- 두 대상의 checkpoint와 임계값을 동결한 뒤 Test를 평가한다. Primary Test는 원래 Test REAL 45 Track 전부와 대상 FAKE만 포함한다: MusicGen REAL45/FAKE45 Track(270 Segment), Udio REAL45/FAKE48 Track(279 Segment). Udio FAKE는 24개 원곡 group에만 있으므로 24-group paired-source 결과는 보조 민감도 분석으로 별도 표시할 수 있다.
- 동일 대상의 네 모델은 동일 Test ID와 공통 지표 함수를 사용한다. Test 점수로 하이퍼파라미터·epoch·임계값을 바꾸지 않는다. Test EER의 교점 임계값은 Test prediction에 적용하지 않는다.

## 남겨 두는 과거 산출물

원래 36개 CNN holdout별 staged search는 사용자 지시로 중단했다. 완료된 trial의 Validation 로그와 checkpoint는 삭제하지 않고 `aborted_search`로 표시한다. 완료된 LR/SVM/MERT holdout별 172개 후보 탐색과 Test 결과 역시 supplementary로 보존한다. 새 primary 결과와 같은 선택 절차의 결과로 섞지 않는다.
