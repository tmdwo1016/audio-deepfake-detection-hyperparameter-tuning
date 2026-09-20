# 이진 탐지 모델링 수정 실행 안내

이 문서는 `modeling_requirements_revised.txt`를 현재 저장소의 새 실험으로 옮긴 실행 안내다. 기존 01–08 단계의 데이터 전처리, EDA, original_audio group split, 10초 segment 및 Track 평균 규칙은 유지한다. 기존 09/13/17 노트북과 README/FINAL_REPORT의 Test 숫자는 **2026-09-13 과거 실험**이다. 새 탐색의 Test 결과로 해석하지 않는다.

## 완료된 실행 (2026-09-20)

Train/Validation 탐색과 공동 최종 Test 평가를 완료했다. Classical 후보는 LR 5개와 SVM 16개, CNN은 Baseline·Stage 1·Stage 2를 합쳐 18개, MERT는 65개를 모두 평가했다. 선택에는 Validation Track EER를 사용했고, 동결 검사 후 네 모델의 Baseline과 Optimized를 같은 Test 1,572 Segment·539 Track에서 평가했다. 실행 출력과 단계별 수치 해석은 [23번 공동 Test 노트북](../archive_notebooks/23_final_binary_test_comparison.ipynb)에 있다. 전체 지표는 [결과 CSV](../results/model_tuning/optimized_test_results.csv), 보고서용 비교는 [최종 보고서](FINAL_REPORT.md) 4.3절에 있다. 최종 실행 ID는 `final_binary_20260919T164939Z`이다.

Optimized Track에서는 MERT가 EER 0.0667, Balanced Accuracy 0.9363으로 가장 높았고, CNN의 AUC 0.984525는 MERT의 0.984480과 매우 가까웠다. SVM의 Test AUC와 EER는 Baseline보다 악화됐고 CNN의 REAL FPR은 0.1111에서 0.2000으로 증가했다. 기존 실험에서 같은 Test의 결과가 공개됐으므로 이를 완전히 미노출인 holdout으로 주장하지 않는다.

과거 2026-09-13 결과에는 실행 당시 기록된 원본 `run_id`가 없다. 보고서의 `historical_20260913`은 신규 실행 `final_binary_20260919T164939Z`와 구분하기 위한 **사후 표기**이며, 기존 CSV의 출처를 새 실행으로 바꾸는 ID가 아니다. 과거 결과 파일은 그대로 보존했다.

## 후속 강건성 평가 완료 (2026-09-20)

사용자가 후속 평가의 실제 실행을 요청해 두 실험을 추가했다. **Unseen generator primary**는 [고정 설정 전이 프로토콜](UNSEEN_FIXED_PROTOCOL.md)을 따른다. MusicGen/Udio 각각 대상 FAKE를 원래 Train·Validation에서 제외하고, Main 실험이 선택한 하이퍼파라미터를 고정한 네 모델을 새로 학습했다. CNN은 대상별로 새로운 가중치와 Train 기준 `pos_weight`를 사용하고, best epoch 및 Segment·Track 임계값만 필터된 Validation으로 선택했다. 결과는 [LR/SVM/MERT 노트북](../archive_notebooks/26_unseen_fixed_classical_mert.ipynb), [CNN 노트북](../archive_notebooks/27_cnn_unseen_fixed_transfer.ipynb), [네 모델 비교 CSV](../results/model_robustness/unseen_fixed_four_model_comparison.csv)에 있다. MusicGen/Udio Track AUC는 LR 0.6178/0.8278, SVM 0.5640/0.8449, CNN 0.7714/0.8551, MERT 0.9407/0.9181이었다. Main 하이퍼파라미터 선택에는 두 대상이 포함된 과거 Validation이 쓰였고 Test 결과도 과거에 공개됐으므로, 이를 완전한 미노출 선택 실험으로 부르지 않는다.

사용자가 중단한 CNN의 36후보 holdout별 탐색은 [중단 기록](../archive_notebooks/14B_cnn_unseen_revised.ipynb)에 남겼다. 완료된 LR/SVM/MERT 172후보 holdout별 재탐색은 [보조 결과](../archive_notebooks/25_unseen_classical_mert.ipynb)로 보존하며 primary와 섞지 않는다.

**Paired MP3**는 [동결된 강건성 프로토콜](ROBUSTNESS_PROTOCOL.md)의 128/64 kbps 설정으로 같은 Test 539곡의 clean·MP3 구간을 맞췄다. 새 paired clean을 두 MP3 조건과 동일한 전체 곡 디코딩 방식으로 다시 추출하고, Main 모델 checkpoint와 원래 clean Validation 임계값을 고정했다. 결과는 [MP3 실행 노트북](../archive_notebooks/24B_optimized_mp3_robustness.ipynb)과 [지표 CSV](../results/model_robustness/mp3/mp3_optimized_20260919T175101Z/metrics.csv)에 있다. Track AUC clean→64 kbps는 LR 0.9429→0.9001, SVM 0.9582→0.9072, CNN 0.9847→0.6839, MERT 0.9845→0.9605였다. CNN의 64 kbps REAL 오탐은 40/45곡이었다. 새 paired clean과 원래 23번 clean은 추출 경로가 달라 별도 QC 표로 차이를 기록했다.

## 당시 실행 순서

1. 프로젝트 루트의 동일 Python 환경에서 [22번 Classical](../archive_notebooks/22_classical_model_tuning.ipynb), [13B CNN](../archive_notebooks/13B_logmel_cnn_tuning.ipynb), [17B MERT](../archive_notebooks/17B_mert_binary_tuning.ipynb) 노트북을 실행했다. 세 노트북은 Train으로 학습하고 Validation만으로 후보, checkpoint, Segment/Track 임계값을 선택했다.
2. 세 노트북의 산출물, manifest hash, cache 설정과 Validation raw score를 확인했다. Classical은 LR 5개·SVM 16개, MERT는 layer 13개 × C 5개인 65개 후보가 모두 기록됐는지 확인했다. CNN은 Stage 1과 Stage 2의 실행 후보 및 Baseline을 기록하고, 완료된 후보 중 반올림 전 Validation Track EER → Track ROC-AUC → 사전 후보 순서로 선택된 설정이 저장된 checkpoint·임계값과 일치하는지 확인했다. MERT의 이전 `v2` cache에는 이 실행에서 요구한 revision/입력/pooling provenance가 없어 새 cache를 만들었다.
3. **모든 선택이 끝난 뒤** [23번 공동 Test 노트북](../archive_notebooks/23_final_binary_test_comparison.ipynb)을 한 번 실행해 각 모델의 Baseline과 Optimized를 동일 Test 집합에서 평가했다. 선택된 Train 모델을 그대로 사용했으며 Train+Validation 재학습은 하지 않았다.
4. 각 노트북의 `### 결과 / 해석 / 주의사항 / 다음 단계` Markdown에는 실제 실행 출력만 옮겨 적고 보고서용 표와 그림을 검토했다. 최종 표에는 각 모델의 선택 설정, REAL·FAKE 표본 수와 비율을 AP와 함께 제시했다. 미실행 셀에는 성능 숫자를 기입하지 않았다.

새 실행의 산출물은 `results/model_tuning/`과 `checkpoints/optimized/`에 저장했다. 기존 `results/baseline/`, `results/cnn/`, `results/mert/` 및 과거 보고서는 덮어쓰지 않았다. 모델별 `run_id`와 manifest hash를 보존했다.

## 공통 평가 규칙

`src/modeling_evaluation.py`를 네 모델 모두가 사용한다. REAL=0, FAKE=1이고 score가 높을수록 FAKE다. LR과 MERT+LR은 `classes_`의 FAKE 열 `predict_proba`, SVM은 `probability=False`의 margin, CNN은 `sigmoid(logit)`를 score로 쓴다. CNN 가중 학습 score나 SVM margin을 보정된 확률로 부르지 않는다. Track score는 동일 track의 Segment score 산술평균이다.

Validation Segment와 Track에서 각각 유한 ROC threshold 후보를 비교하여 `|FPR−FNR|`, HTER, 높은 threshold 순으로 임계값을 정한다. EER은 ROC의 `FPR−FNR` 부호가 바뀌는 구간을 선형 보간한 분리력 지표다. Test 분류 지표와 HTER에는 **저장된 Validation 임계값만** 적용한다. Test EER의 교점은 Test 분류에 사용하지 않는다.

ROC-AUC는 모든 threshold에서의 순위화, AP는 `average_precision_score`의 비보간 요약값이다. FAKE AP와 REAL AP 및 각 class 비율을 함께 기록한다. Balanced Accuracy와 Macro-F1은 두 class에 같은 중요도를 주며, REAL FPR은 REAL을 AI로 오탐한 비율, FAKE Miss Rate는 FAKE를 REAL로 놓친 비율이다. Segment와 Track 지표를 모두 저장한다. 한 class만 있으면 정의되지 않은 지표를 NaN과 사유로 표시한다. score 결측, 중복 segment ID, track 내부 label 불일치는 오류로 중단한다.

## 확인된 입력 상태와 한계

현재 `data/metadata/segment_manifest_10s.csv`에서 segment 수는 Train 6,967, Validation 1,538, Test 1,572이며 원본 group은 서로 겹치지 않는다. Train segment REAL 621 / FAKE 6,346, Validation REAL 132 / FAKE 1,406, Test REAL 135 / FAKE 1,437이다. 이 숫자는 manifest 검사 결과이며 **새 모델 성능은 아니다**. 동일 seed=42도 장비와 라이브러리가 다르면 완전 동일 결과를 보장하지 않는다.

FMA REAL과 Echoes TTA FAKE의 차이에는 생성 여부 외에 출처·장르·코덱·제작 방식이 섞일 수 있다. MERT는 외부 사전학습 정보가 있으므로 CNN의 처음부터 학습한 결과와 같은 학습 정보량으로 비교할 수 없다. 사전학습 데이터와 평가 데이터의 중복 여부를 확인하지 못하면 그 한계를 결과 해석에 남긴다.

## 후속 분석 연결

새 이진 모델, 고정 설정 unseen, paired MP3, 과거 attribution 결과를 `run_id`로 구분해 연결했다. 이번 unseen 평가의 대상 FAKE는 재학습 Train/Validation에서 제외했지만, 고정한 설정은 과거 두 대상이 포함된 Validation에서 선택됐다. 대상 생성기를 모델 설정 선택 단계에서도 제외해야 완전히 미노출인 선택 절차를 평가할 수 있다. MP3 비교에는 같은 Test REAL/FAKE track의 clean/변환 쌍과 clean Validation 임계값을 사용했다. 12-way attribution은 실제 class 목록을 확인한 별도 다중분류 실험이며, 이진 EER/BCE 규칙을 그대로 적용하지 않는다. Genre, acoustic feature, PCA/UMAP 및 silhouette 해석은 탐색적 분석으로 표시한다.
