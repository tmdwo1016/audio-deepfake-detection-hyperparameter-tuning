# 변경사항

## 2026-09-20 — 실행 기록을 제출본으로 사용

- 별도 설명용 `notebooks/` 9개를 제거하고, 저장 출력이 있는 `archive_notebooks/` 32개를 제출 기준으로 삼았다.
- 보관 노트북의 반복적인 범용 주석과 대화용 Markdown을 다듬었다. 계산 코드, 저장 출력, 기존 결과 표와 그림은 유지했다.
- 저장된 공동 Test CSV에서 LR·RBF-SVM·Log-Mel CNN·Frozen MERT+LR의 Baseline/Optimized 비교 그림을 모델별 한 장씩 만들었다. 학습·튜닝·Test 추론은 다시 실행하지 않았다.
- README·제출 안내·실험 기록 목록·제출 ZIP을 32개 실행 기록 기준으로 고쳤다.

## 2026-09-20 — 제출 노트북 9개로 정리

- 연구 과정의 실험 노트북 32개를 `archive_notebooks/`에 보관했다. 이후 요청에 따라 계산 코드와 저장 출력을 유지한 채 주석·마크다운을 간결하게 수정했다.
- 저장된 manifest, 튜닝 표, 예측, 지표, 그림을 바탕으로 `notebooks/01`–`09`를 새로 작성했다. 데이터 준비에서 최종 해석까지 읽는 순서로 배치했다.
- 각 노트북에 핵심 코드의 한국어 주석과 결과 출처를 넣었다. 새 코드 셀은 실행하지 않았고 출력도 만들어 넣지 않았다.
- 당시 [대응표](docs/EXPERIMENT_INDEX.md), [제출 안내](docs/SUBMISSION_GUIDE.md), README와 검토용 ZIP의 목록을 9개 구조에 맞췄다. 모델 학습·튜닝·추론·평가와 checkpoint·prediction·metric 생성은 다시 하지 않았다.
- 이진 탐지 본표를 Track EER·ROC-AUC·REAL FPR·FAKE miss 네 지표로 통일했다. 다른 지표와 Segment 결과는 기존 CSV에 보존하고 제출 ZIP을 다시 만들었다.
- 제출 노트북 01~09의 반복 소제목과 긴 Markdown을 줄이고, 데이터 분할·표준화·모델 점수·CNN 학습 순서를 설명하는 한국어 코드 주석을 보강했다. 계산 코드, 저장 출력, 표와 그림은 유지했다.

## 2026-09-20 — 최신 모델링·강건성 및 제출 정리

- 네 이진 모델을 Train/Validation에서 선택하고, 동결한 설정으로 같은 Test에서 Baseline·Optimized를 평가했다. 결과는 `23_final_binary_test_comparison.ipynb`와 `results/model_tuning/`에 저장했다.
- MusicGen/Udio 대상 FAKE를 Train/Validation에서 제거하고 Main 설정을 고정한 fresh fit을 실행했다. `26_unseen_fixed_classical_mert.ipynb`, `27_cnn_unseen_fixed_transfer.ipynb`가 primary이며, `25_unseen_classical_mert.ipynb`는 대상별 재탐색 보조 결과다. CNN의 36후보 탐색은 사용자 지시에 따라 중단하고 `14B_cnn_unseen_revised.ipynb`에 기록했다.
- 네 Optimized 모델의 clean·MP3 128·64 kbps paired 평가를 `24B_optimized_mp3_robustness.ipynb`와 `results/model_robustness/mp3/`에 저장했다.
- 제출용 [자료 안내](docs/SUBMISSION_GUIDE.md)를 추가하고 README의 과거 결과 중복을 줄였다. 전체 노트북 코드 셀의 단계 주석과 새 핵심 노트북의 실험 설명을 보강했다. Python 코드와 노트북 코드의 서식을 정리하며 계산 로직과 저장된 출력을 유지했다.
- `AI_music_project_submission_20260920.zip`은 현재 보고서·코드·노트북·작은 결과·metadata를 담은 검토용 묶음이다. 대용량 원본 audio/cache/checkpoint는 포함하지 않는다.

## 2026-09-13 — 분석 최종본

### 문서

- 프로젝트 목적, 분석 흐름, 대표 결과, 실행 순서를 담은 `README.md` 추가
- 방법론, 전체 결과, 해석, 제한사항, 후속 연구를 담은 `docs/FINAL_REPORT.md` 추가
- 모든 완료 노트북의 마지막 마크다운을 실제 실행 결과 기준으로 갱신
- GitHub에서 바로 확인할 수 있도록 결과 그림과 상대 경로 연결
- 원본 manifest부터 waveform, handcrafted, Log-Mel, MERT까지의 schema와 tensor shape를 설명하는 `docs/DATA_PREPROCESSING.md` 추가
- Group split, duration 정규화, segment 위치 공식, padding 현황, 결측값 의미, strict-balanced 구성과 단계별 QC 상세화

### 데이터 및 전처리

- Echoes TTA 충돌 및 누락 검증
- 296개 FMA REAL track 매칭·다운로드·decode QC
- 3,458-track master manifest 구축
- `original_audio` 기준 group split으로 source-family 누수 방지
- 10,077개 10초 segment와 266-D handcrafted feature 생성

### Detection

- Logistic Regression 및 RBF-SVM baseline 추가
- Log-Mel CNN 학습과 Segment/Track 평가 추가
- Frozen MERT-v1-95M 13-layer 탐색 및 LR baseline 추가
- MERT/Transformers 및 Apple Silicon 실행 안정성을 반영한 `17_mert_frozen_baseline_v2.ipynb` 추가

### Robustness

- MusicGen/Udio unseen-generator 실험 추가
- MP3 128/64 kbps full-decode robustness 실험 추가
- Validation threshold 고정 및 Train-only preprocessing 원칙 적용

### Generator fingerprint

- Handcrafted + RBF-SVM 12-way generator attribution 추가
- Source coverage를 통제한 controlled 및 strict-balanced 실험 추가
- Frozen MERT Layer 10 + LR attribution 추가
- Handcrafted/MERT PCA·UMAP 및 generator/genre silhouette 분석 추가
- Generator × genre recall과 Train-only acoustic feature ANOVA 추가

### 주요 결과

- In-domain Track ROC-AUC: RBF-SVM 0.9644, CNN 0.9772, MERT 0.9872
- MP3 64 kbps Track ROC-AUC: RBF-SVM 0.9062, CNN 0.8256
- Strict-balanced 12-way Macro-F1: Handcrafted 0.7447, MERT 0.8980
- 높은 supervised attribution 성능과 낮은 silhouette를 함께 확인

### 저장 정책

- 재현 가능한 compact CSV/JSON/PNG 결과를 `results/`에 보존
- 원본 오디오, 대용량 feature cache, model checkpoint는 `.gitignore`로 제외
