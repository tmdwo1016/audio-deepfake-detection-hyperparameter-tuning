# 제출 자료 안내

## 읽는 순서

제출 노트북은 [archive_notebooks](../archive_notebooks/)의 실행 기록 32개다. 전체 목록은 [실험 기록 안내](EXPERIMENT_INDEX.md)에 있다. 먼저 [README](../README.md)와 [제출용 Word 최종보고서 수정본](AI_생성음악_탐지_모델링_최종보고서_수정본.docx)으로 연구 질문과 결과를 보고, 다음 기록을 순서대로 읽으면 된다.

| 단계 | 실행 기록 | 확인할 내용 |
|---|---|---|
| 데이터와 특징 | [05 분할](../archive_notebooks/05_group_split.ipynb), [07 구간](../archive_notebooks/07_build_segment_manifest.ipynb), [08 특징](../archive_notebooks/08_extract_handcrafted_features.ipynb) | 원곡 단위 분할과 모델 입력 |
| 모델 선택 | [22 LR·SVM](../archive_notebooks/22_classical_model_tuning.ipynb), [13B CNN](../archive_notebooks/13B_logmel_cnn_tuning.ipynb), [17B MERT](../archive_notebooks/17B_mert_binary_tuning.ipynb) | Train 학습·Validation 설정 선택; 각 파일 끝에 Validation과 Test 비교표·그림 |
| 공동 Test | [23 네 모델 비교](../archive_notebooks/23_final_binary_test_comparison.ipynb) | Baseline·Optimized 지표와 모델별 비교 그림 |
| 강건성 | [26 LR·SVM·MERT](../archive_notebooks/26_unseen_fixed_classical_mert.ipynb), [27 CNN](../archive_notebooks/27_cnn_unseen_fixed_transfer.ipynb), [24B MP3](../archive_notebooks/24B_optimized_mp3_robustness.ipynb) | 생성기 제외 후 고정 설정 전이, paired MP3 |
| 생성기 분류·해석 | [18B 균형 통제](../archive_notebooks/18B_balanced_controlled_generator_attribution.ipynb), [19 MERT 분류](../archive_notebooks/19_generator_attribution_mert.ipynb), [20 v2 시각화](../archive_notebooks/20_generator_fingerprint_pca_umap_v2.ipynb), [21 장르·특징](../archive_notebooks/21_generator_genre_feature_analysis.ipynb) | 12종 분류와 원곡·장르 영향 |

기존 학습 코드와 저장 출력은 유지했다. 22·13B·17B의 끝에 저장된 Validation 후보 CSV와 공동 Test 결과 CSV를 읽는 비교 셀을 추가해 실행했다. 모델 재학습·재튜닝·Test 재평가는 하지 않았다. 14B CNN 대상별 후보 탐색은 중단된 기록이고, 25번 LR·SVM·MERT 대상별 재탐색은 보조 분석이다. Unseen 결과의 주실험은 Main 하이퍼파라미터를 고정한 26·27번이다.

## 결과 파일과 그림

| 결과 | 원본 |
|---|---|
| Validation 후보·Baseline 비교 | [LR](../results/model_tuning/validation_comparisons/logistic_validation_comparison.png), [SVM](../results/model_tuning/validation_comparisons/svm_validation_comparison.png), [CNN](../results/model_tuning/validation_comparisons/cnn_validation_comparison.png), [MERT](../results/model_tuning/validation_comparisons/mert_validation_comparison.png) |
| 공동 Test Baseline 비교 | [LR](../results/model_tuning/test_comparisons/logistic_test_comparison.png), [SVM](../results/model_tuning/test_comparisons/svm_test_comparison.png), [CNN](../results/model_tuning/test_comparisons/cnn_test_comparison.png), [MERT](../results/model_tuning/test_comparisons/mert_test_comparison.png) |
| 공동 Test | [네 모델 지표](../results/model_tuning/optimized_test_results.csv), [Baseline 변화량](../results/model_tuning/baseline_vs_optimized.csv) |
| 튜닝 전후 그림 | [LR](../results/model_tuning/model_comparisons/logistic_regression_baseline_vs_optimized.png), [SVM](../results/model_tuning/model_comparisons/rbf_svm_baseline_vs_optimized.png), [CNN](../results/model_tuning/model_comparisons/logmel_cnn_baseline_vs_optimized.png), [MERT](../results/model_tuning/model_comparisons/frozen_mert_lr_baseline_vs_optimized.png) |
| MusicGen·Udio | [고정 설정 네 모델 비교](../results/model_robustness/unseen_fixed_four_model_comparison.csv), [실험 절차](UNSEEN_FIXED_PROTOCOL.md) |
| MP3 | [조건별 지표](../results/model_robustness/mp3/mp3_optimized_20260919T175101Z/metrics.csv), [paired 변화량](../results/model_robustness/mp3/mp3_optimized_20260919T175101Z/paired_deltas.csv) |
| 생성기 분류 | [수작업 특징·MERT 비교](../results/generator_attribution/mert/handcrafted_vs_mert_generator_attribution.csv) |

REAL=0, FAKE=1이다. 곡 점수는 같은 곡의 10초 구간 점수 평균이다. 이진 탐지 본표의 EER과 ROC-AUC는 Test 점수의 분리력을 나타내며, REAL FPR과 FAKE miss에는 Validation에서 정한 임계값을 고정해 적용했다. Test 점수로 모델이나 임계값을 다시 고르지 않았다.

Main 하이퍼파라미터 선택에는 MusicGen·Udio가 포함된 과거 Validation이 사용됐고, 과거 Test 결과도 이미 공개됐다. 따라서 후속 unseen 분석을 모든 선택 단계까지 미노출인 평가로 해석하지 않는다. 실험 전에 고정한 프로토콜 문서 두 개에는 결과 메타데이터와 연결된 SHA-256이 있어 당시 경로 표기를 그대로 남겼다.

## 제출 묶음

`AI_music_project_submission_20260920.zip`에는 실행 기록 32개, 설명 문서, 코드와 작은 결과 파일이 들어 있다. `SUBMISSION_MANIFEST.json`에 포함 파일의 SHA-256을 기록했다. 원본 오디오·대용량 특징 cache·checkpoint는 포함하지 않는다. 내용을 수정했을 때는 `python scripts/build_submission_bundle.py`로 ZIP을 다시 만든다.
