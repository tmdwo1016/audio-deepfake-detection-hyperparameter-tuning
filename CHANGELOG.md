# 변경사항

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
