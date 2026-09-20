# 실험 기록 안내

제출 노트북은 [archive_notebooks](../archive_notebooks/)의 실행 기록 32개다. 번호는 작업 순서이며, B와 v2는 같은 주제를 다시 다룬 기록이다. 지금 모델 비교의 기준은 22·13B·17B에서 설정을 고른 뒤 23번에서 같은 Test를 평가한 실행이다. 14B는 중단된 탐색, 25번은 대상별 재탐색 보조 결과다.

| 실행 기록 | 내용 |
|---|---|
| [01_data_quality_check](../archive_notebooks/01_data_quality_check.ipynb) | Echoes 데이터 품질과 TTA 선택 |
| [02_fma_matching_check](../archive_notebooks/02_fma_matching_check.ipynb) | REAL 원곡 매칭과 생성기별 원곡 구조 |
| [03_fma_audio_validation](../archive_notebooks/03_fma_audio_validation.ipynb) | FMA 오디오 유효성 확인 |
| [04_build_master_manifest](../archive_notebooks/04_build_master_manifest.ipynb) | REAL/FAKE master manifest |
| [05_group_split](../archive_notebooks/05_group_split.ipynb) | `original_audio` 그룹 분할 |
| [06_eda](../archive_notebooks/06_eda.ipynb) | 클래스·생성기·장르 분포 |
| [07_build_segment_manifest](../archive_notebooks/07_build_segment_manifest.ipynb) | 10초 Segment 규칙과 입력 ID |
| [08_extract_handcrafted_features](../archive_notebooks/08_extract_handcrafted_features.ipynb) | 266차원 수작업 특징 정의 |
| [09_baseline_models](../archive_notebooks/09_baseline_models.ipynb) | 초기 Logistic Regression·RBF-SVM 학습과 Segment·Track 평가 |
| [10_baseline_subgroup_analysis](../archive_notebooks/10_baseline_subgroup_analysis.ipynb) | 초기 subgroup 오류 분석 |
| [11_unseen_generator_handcrafted](../archive_notebooks/11_unseen_generator_handcrafted.ipynb) | 초기 unseen 비교 기록 |
| [12_mp3_robustness_handcrafted](../archive_notebooks/12_mp3_robustness_handcrafted.ipynb) | 초기 handcrafted MP3 기록 |
| [13_logmel_cnn](../archive_notebooks/13_logmel_cnn.ipynb) | Log-Mel 정의와 초기 CNN 기록 |
| [13B_logmel_cnn_tuning](../archive_notebooks/13B_logmel_cnn_tuning.ipynb) | CNN Baseline·18 trial·best checkpoint 선택; 마지막에 Validation·Test 비교 |
| [14_cnn_unseen_generator](../archive_notebooks/14_cnn_unseen_generator.ipynb) | 초기 CNN unseen 기록 |
| [14B_cnn_unseen_revised](../archive_notebooks/14B_cnn_unseen_revised.ipynb) | 중단된 CNN 후보 탐색 기록만 |
| [15_cnn_mp3_robustness](../archive_notebooks/15_cnn_mp3_robustness.ipynb) | 초기 CNN MP3 기록 |
| [16_final_integration_analysis](../archive_notebooks/16_final_integration_analysis.ipynb) | 초기 SVM·CNN 중심 비교와 연구 질문 정리; LR 결과는 09번 |
| [17_mert_frozen_baseline_v2](../archive_notebooks/17_mert_frozen_baseline_v2.ipynb) | MERT 표현과 초기 Baseline 기록 |
| [17B_mert_binary_tuning](../archive_notebooks/17B_mert_binary_tuning.ipynb) | MERT layer×C 65후보 선택; 마지막에 Validation·Test 비교 |
| [18_generator_attribution_handcrafted_v3](../archive_notebooks/18_generator_attribution_handcrafted_v3.ipynb) | 수작업 특징의 12종 분류 |
| [18B_balanced_controlled_generator_attribution](../archive_notebooks/18B_balanced_controlled_generator_attribution.ipynb) | source·class 통제 분석 |
| [19_generator_attribution_mert](../archive_notebooks/19_generator_attribution_mert.ipynb) | MERT의 12종 분류 |
| [20_generator_fingerprint_pca_umap](../archive_notebooks/20_generator_fingerprint_pca_umap.ipynb) | 초기 PCA·UMAP 기록 |
| [20_generator_fingerprint_pca_umap_v2](../archive_notebooks/20_generator_fingerprint_pca_umap_v2.ipynb) | 최종 PCA·UMAP·silhouette |
| [21_generator_genre_feature_analysis](../archive_notebooks/21_generator_genre_feature_analysis.ipynb) | 장르별 성능과 음향 특징 |
| [22_classical_model_tuning](../archive_notebooks/22_classical_model_tuning.ipynb) | LR 5개·SVM 16개 후보 선택; 마지막에 모델별 Validation·Test 비교 |
| [23_final_binary_test_comparison](../archive_notebooks/23_final_binary_test_comparison.ipynb) | 동결된 네 모델의 공동 Test |
| [24B_optimized_mp3_robustness](../archive_notebooks/24B_optimized_mp3_robustness.ipynb) | paired MP3 평가 |
| [25_unseen_classical_mert](../archive_notebooks/25_unseen_classical_mert.ipynb) | 대상별 재탐색 보조 결과 |
| [26_unseen_fixed_classical_mert](../archive_notebooks/26_unseen_fixed_classical_mert.ipynb) | LR/SVM/MERT 고정 설정 unseen primary |
| [27_cnn_unseen_fixed_transfer](../archive_notebooks/27_cnn_unseen_fixed_transfer.ipynb) | CNN 고정 설정 unseen primary |
