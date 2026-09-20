# AI 생성 음악 탐지와 생성기 특성 분석

실제 음악과 AI 생성 음악을 구별하고, 처음 보는 생성기와 MP3 압축에서도 탐지가 가능한지 살펴본 기계학습·딥러닝 수업 프로젝트다. 마지막에는 AI 음악이 어느 생성기로 만들어졌는지도 분석한다.

## 실험 기록 읽는 순서

제출 노트북은 [archive_notebooks](archive_notebooks/)의 실행 기록 32개다. 데이터 준비부터 후속 분석까지 모두 이 폴더에 있으며, 각 코드 셀의 저장 출력도 남아 있다. 처음부터 전부 읽기보다 아래 순서로 주요 기록을 보면 된다. 전체 목록은 [실험 기록 안내](docs/EXPERIMENT_INDEX.md)에 있다.

| 단계 | 기록 | 확인할 내용 |
|---|---|---|
| 데이터·입력 | [05 원곡 분할](archive_notebooks/05_group_split.ipynb), [07 구간](archive_notebooks/07_build_segment_manifest.ipynb), [08 특징](archive_notebooks/08_extract_handcrafted_features.ipynb) | 원곡 누수 방지, 10초 구간, 266차원 특징 |
| 초기 모델 | [09 LR·SVM](archive_notebooks/09_baseline_models.ipynb) | 두 모델의 초기 학습과 Segment·Track 결과 |
| 모델 선택 | [22 LR·SVM](archive_notebooks/22_classical_model_tuning.ipynb), [13B CNN](archive_notebooks/13B_logmel_cnn_tuning.ipynb), [17B MERT](archive_notebooks/17B_mert_binary_tuning.ipynb) | 각 파일 끝에 Validation 후보 비교와 공동 Test 비교표·그림 |
| 공동 Test | [23 네 모델 비교](archive_notebooks/23_final_binary_test_comparison.ipynb) | Baseline·Optimized 결과와 모델별 비교 그림 |
| 강건성 | [26 Classical·MERT unseen](archive_notebooks/26_unseen_fixed_classical_mert.ipynb), [27 CNN unseen](archive_notebooks/27_cnn_unseen_fixed_transfer.ipynb), [24B MP3](archive_notebooks/24B_optimized_mp3_robustness.ipynb) | 고정 설정 전이와 같은 곡의 압축 전후 |
| 생성기 분류·해석 | [18B 균형 통제](archive_notebooks/18B_balanced_controlled_generator_attribution.ipynb), [19 MERT 분류](archive_notebooks/19_generator_attribution_mert.ipynb), [20 v2 시각화](archive_notebooks/20_generator_fingerprint_pca_umap_v2.ipynb), [21 장르·특징](archive_notebooks/21_generator_genre_feature_analysis.ipynb) | 12종 분류, PCA·UMAP, 특징 차이 |

## 데이터와 평가 순서

Echoes TTA의 AI 생성 음악과 대응되는 FMA 실제 음악을 정리해 3,458곡(FAKE 3,162곡, REAL 296곡)을 사용했다. 같은 `original_audio`에서 나온 모든 파일과 구간을 같은 Train·Validation·Test에 두어 원곡이 분할 사이에 섞이지 않도록 했다. 총 10,077개의 10초 구간을 만들었고, 모델의 입력에 따라 266차원 특징, Log-Mel 또는 MERT 표현을 사용했다. 데이터 처리 규칙은 [전처리 설명](docs/DATA_PREPROCESSING.md)에 있다.

모델과 하이퍼파라미터는 Train·Validation으로 정하고, 분류 임계값도 Validation에서 선택했다. 그 설정을 고정한 뒤 공동 Test 539곡(REAL 45곡, FAKE 494곡)을 평가했다. 아래는 **곡 단위 Optimized 결과**다. 수치는 [저장된 공동 Test CSV](results/model_tuning/optimized_test_results.csv)에서 가져왔다.

| 모델 | EER ↓ | ROC-AUC ↑ | REAL FPR ↓ | FAKE miss ↓ |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.1333 | 0.9419 | 0.1333 | 0.0931 |
| RBF-SVM | 0.1255 | 0.9586 | 0.1556 | 0.0445 |
| Log-Mel CNN | 0.0749 | 0.9845 | 0.2000 | 0.0243 |
| Frozen MERT + LR | 0.0667 | 0.9845 | 0.0889 | 0.0385 |

CNN과 MERT의 AUC는 소수 넷째 자리까지 같지만, MERT의 EER과 REAL FPR이 더 낮았다. CNN은 FAKE를 놓치는 비율이 낮은 대신 REAL 오탐이 상대적으로 많았다. EER·AUC는 Test 점수에서 계산했고, 두 오류율은 Validation에서 정한 임계값을 적용했다. 모델별 Baseline 비교와 그림은 [23번 공동 Test 기록](archive_notebooks/23_final_binary_test_comparison.ipynb)에 있다.

| LR | RBF-SVM |
|---|---|
| ![LR의 튜닝 전후 Test 비교](results/model_tuning/model_comparisons/logistic_regression_baseline_vs_optimized.png) | ![SVM의 튜닝 전후 Test 비교](results/model_tuning/model_comparisons/rbf_svm_baseline_vs_optimized.png) |

| Log-Mel CNN | Frozen MERT + LR |
|---|---|
| ![CNN의 튜닝 전후 Test 비교](results/model_tuning/model_comparisons/logmel_cnn_baseline_vs_optimized.png) | ![MERT의 튜닝 전후 Test 비교](results/model_tuning/model_comparisons/frozen_mert_lr_baseline_vs_optimized.png) |

그림의 막대는 Optimized−Baseline의 퍼센트포인트 변화량이다. 오른쪽 숫자는 실제 전후 값이며, REAL FPR과 FAKE miss에는 각 설정의 Validation 임계값을 적용했다.

## 강건성과 생성기 분류

MusicGen과 Udio를 각각 Train·Validation에서 제외하고, Main 실험에서 고른 모델 설정을 고정한 채 해당 Train으로 가중치를 새로 학습한 **기존 실험**을 분석했다. 두 대상의 곡 단위 AUC는 CNN 0.7714/0.8551, MERT 0.9407/0.9181이었다(MusicGen/Udio 순서). 이는 대상별 하이퍼파라미터를 다시 찾은 결과가 아니다. 다만 Main 설정을 고를 당시 두 생성기는 원래 Validation에 포함됐으므로, 모델 설정까지 완전히 미노출이었던 실험으로 해석하지 않는다. [고정 설정 비교 CSV](results/model_robustness/unseen_fixed_four_model_comparison.csv)와 [프로토콜](docs/UNSEEN_FIXED_PROTOCOL.md)에 범위를 적었다.

같은 Test 곡을 clean·MP3 128·64 kbps로 대응시킨 저장 결과에서는 64 kbps의 CNN AUC가 0.6839로 내려갔고 MERT는 0.9605였다. 압축 전후 차이는 같은 디코딩 경로에서 만든 **paired clean**을 기준으로 계산했다. [MP3 결과](results/model_robustness/mp3/mp3_optimized_20260919T175101Z/metrics.csv)에 다른 지표도 있다.

12종 생성기 분류에서는 전체 FAKE 곡과 원곡·생성기 구성을 엄격하게 맞춘 집합을 따로 봤다. 균형 통제 후에도 분류 성능이 남았지만, 차원 축소 그림과 silhouette만으로 생성기마다 깔끔한 군집이 생긴다고 말할 수는 없다. 수치와 해석은 [19번 분류](archive_notebooks/19_generator_attribution_mert.ipynb), [20번 시각화](archive_notebooks/20_generator_fingerprint_pca_umap_v2.ipynb), [21번 특징 분석](archive_notebooks/21_generator_genre_feature_analysis.ipynb)에 있다.

## 제출 파일과 기록

- [제출 자료 안내](docs/SUBMISSION_GUIDE.md): 읽는 순서, 자료 범위, 마지막 확인 사항
- [제출용 Word 최종보고서 수정본](docs/AI_생성음악_탐지_모델링_최종보고서_수정본.docx): 모델별 Validation·Test 비교와 강건성·생성기 분류 결과
- [상세 Markdown 보고서](docs/FINAL_REPORT.md): 실험 방법, 수치, 한계
- [실험 기록 안내](docs/EXPERIMENT_INDEX.md): 보관 노트북 32개의 순서와 역할
- `AI_music_project_submission_20260920.zip`: 노트북·문서·작은 결과를 담은 검토용 묶음. 포함 파일의 SHA-256은 ZIP의 `SUBMISSION_MANIFEST.json`에 있다.

원본 오디오, 대용량 cache와 checkpoint는 제출 ZIP에 넣지 않았다. ZIP 안의 실행 기록에는 저장 출력이 있으며, 전체 학습을 처음부터 재현하려면 로컬 원본 자료가 필요하다. 과거 `AI_music_project_package_v3_notebook_based` 묶음은 이전 제출 스냅샷이다.
