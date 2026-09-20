# 새 이진 모델 강건성 실험 프로토콜

2026-09-20에 고정한 후속 실험 계획이다. [새 이진 모델 공동 Test](../23_final_binary_test_comparison.ipynb)의 수치를 본 뒤 작성했으므로, 같은 Test의 기존 공개 이력과 새 설계의 한계를 결과에서 명시한다. 아래 조건은 후속 실험의 Test 점수를 계산하기 전에 고정했다.

## Unseen generator

- 대상은 기존 후속 실험과 연결되는 `musicgen`, `udio` 두 생성기다. 각각 독립 실험으로 다룬다.
- 기존 `original_audio` Train/Validation/Test group split과 10초 Segment 경계를 바꾸지 않는다. 대상 생성기의 FAKE 행만 Train과 Validation에서 제거하며, 그 행을 Test로 옮기지 않는다. REAL과 다른 생성기 행은 원래 split에 남긴다.
- 대상이 원래 탐색의 Validation에 있었으므로 22/13B/17B에서 선택한 하이퍼파라미터를 그대로 가져와 ‘진정한 unseen 선택’이라고 부르지 않는다. 각 대상별로 LR의 C 5개, SVM의 C×gamma 16개, MERT의 layer×C 65개, CNN의 Baseline과 staged 후보를 **필터된 Train/Validation에서 새로** 학습·선택한다. 선택 순서는 반올림 전 Validation Track EER 최소 → ROC-AUC 최대 → 사전 후보 순서다. 모델과 Segment·Track 임계값을 동결한 뒤 Test를 평가한다.
- Primary Test는 기존 Test의 모든 REAL 45 Track과 해당 생성기의 FAKE Track만 사용한다. MusicGen은 REAL 45·FAKE 45 Track(270 Segment), Udio는 REAL 45·FAKE 48 Track(279 Segment)이다. Udio FAKE는 24개 원곡 group에만 있으며 REAL 45개 중 21개는 Udio FAKE와 짝이 없다는 한계를 기록한다. 필요하면 이 24개 원곡에 한정한 paired-source 결과를 보조 분석으로 별도 표시한다.
- 네 모델 모두 같은 primary Test ID에 공통 평가 함수를 사용하고, 저장된 대상별 Validation 임계값으로 Test 분류 지표를 구한다. Test EER 임계값으로 예측을 조정하지 않는다.

## Paired MP3

- 모델은 [공동 Test](../23_final_binary_test_comparison.ipynb)에서 동결된 네 **Optimized** 설정을 사용한다. 재학습·재튜닝하지 않는다.
- bitrate는 기존 프로토콜의 128 kbps와 64 kbps로 고정한다. 기존 Test REAL·FAKE 539 Track 모두에서 clean과 두 MP3 파일을 동일 `track_sample_id`로 대응시킨다. 원래 Segment 시작·종료 시각과 10초 경계를 유지한다.
- Clean과 두 압축 조건 모두 동일한 **track 전체 decode → 원래 Segment 시각으로 slice** 방식으로 각 표현을 추출한다. 과거 공동 Test의 clean score는 새 paired clean 추출과 비교하는 QC 기준이며, 추출 방식 때문에 차이가 있으면 paired 분석의 분모로 쓰지 않는다. Segment와 Track 분류에는 원래 clean Validation에서 저장된 각 모델·수준의 임계값을 그대로 적용한다.
- Clean 대비 같은 ID의 AUC, 두 AP, EER, Balanced Accuracy, Macro-F1, REAL FPR, FAKE Miss Rate, HTER 변화와 오류 방향을 기록한다. 압축 후 점수를 사용해 모델이나 임계값을 고르지 않는다.

## 공통 해석 범위

과거 2026-09-13 unseen/MP3 Test 결과와 이번 결과는 서로 다른 학습 실행이다. 과거 결과 노출은 새 search space, 대상 생성기 및 bitrate 선택에 영향을 줬을 수 있다. 후속 결과를 완전히 사전 등록된 미노출 holdout의 추정치라고 표현하지 않는다. MERT의 외부 사전학습 데이터 중복 여부는 확인되지 않았다.
