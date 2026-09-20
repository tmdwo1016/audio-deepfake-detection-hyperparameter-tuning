"""Behavioral checks for the shared Validation/Test evaluation contract."""

import unittest

import numpy as np
import pandas as pd

from src.modeling_evaluation import (
    aggregate_track_scores,
    evaluate_binary_predictions,
    select_best_candidate,
    select_validation_threshold,
)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "segment_id": ["a", "b", "c", "d"],
            "track_id": ["real", "real", "fake", "fake"],
            "original_audio_id": ["source_a", "source_a", "source_b", "source_b"],
            "label": [0, 0, 1, 1],
            "score": [0.1, 0.3, 0.7, 0.9],
        })

    def test_track_mean_and_separate_thresholds(self):
        result = evaluate_binary_predictions(self.frame)
        self.assertEqual(result["track_scores"]["score"].tolist(), [0.2, 0.8])
        self.assertEqual(result["thresholds"]["segment"], 0.7)
        self.assertEqual(result["thresholds"]["track"], 0.8)
        self.assertEqual(result["track"]["confusion_matrix"], [[1, 0], [0, 1]])

    def test_fixed_test_threshold_is_used_for_prediction(self):
        result = evaluate_binary_predictions(
            self.frame, thresholds={"segment": 0.95, "track": 0.95}
        )
        self.assertEqual(result["segment"]["fake_miss_rate"], 1.0)
        self.assertEqual(result["segment"]["eer"], 0.0)
        self.assertEqual(result["segment"]["hter"], 0.5)

    def test_interpolated_eer_and_finite_threshold(self):
        frame = pd.DataFrame({
            "segment_id": ["a", "b", "c"], "track_id": ["a", "b", "c"],
            "original_audio_id": ["a", "b", "c"], "label": [0, 0, 1],
            "score": [0.9, 0.1, 0.8],
        })
        result = evaluate_binary_predictions(frame)
        self.assertAlmostEqual(result["segment"]["eer"], 0.5)
        self.assertTrue(np.isfinite(select_validation_threshold(frame.label, frame.score)))

    def test_bad_ids_and_track_labels_stop_evaluation(self):
        duplicate = self.frame.copy()
        duplicate.loc[1, "segment_id"] = "a"
        with self.assertRaisesRegex(ValueError, "Duplicate segment_id"):
            aggregate_track_scores(duplicate)
        mixed = self.frame.copy()
        mixed.loc[1, "label"] = 1
        with self.assertRaisesRegex(ValueError, "both REAL and FAKE"):
            evaluate_binary_predictions(mixed)

    def test_candidate_tie_uses_predeclared_order(self):
        rows = [
            {"track_eer": 0.2, "track_roc_auc": 0.9, "candidate_order": 2},
            {"track_eer": 0.2, "track_roc_auc": 0.9, "candidate_order": 0},
        ]
        self.assertEqual(select_best_candidate(rows)["candidate_order"], 0)


if __name__ == "__main__":
    unittest.main()
