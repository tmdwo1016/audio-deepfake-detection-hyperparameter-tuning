"""Final Test must cover the fixed manifest and keep Validation thresholds."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.modeling_final import evaluate_frozen_test_run, verify_test_score_frame


class FinalTestContract(unittest.TestCase):
    def setUp(self):
        self.manifest = pd.DataFrame({
            "segment_id": ["r1", "r2", "f1", "f2"],
            "track_sample_id": ["r", "r", "f", "f"],
            "original_audio": ["source_r", "source_r", "source_f", "source_f"],
            "label_id": [0, 0, 1, 1], "split": ["test"] * 4,
        })
        self.scores = pd.DataFrame({
            "segment_id": ["r1", "r2", "f1", "f2"],
            "track_id": ["r", "r", "f", "f"],
            "original_audio_id": ["source_r", "source_r", "source_f", "source_f"],
            "label": [0, 0, 1, 1], "score": [0.1, 0.2, 0.8, 0.9],
        })

    def test_missing_or_relabelled_test_segment_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Test segment mismatch"):
            verify_test_score_frame(self.manifest, self.scores.iloc[:-1])
        altered = self.scores.copy()
        altered.loc[:1, "original_audio_id"] = "different"
        with self.assertRaisesRegex(ValueError, "original_audio_id differs"):
            verify_test_score_frame(self.manifest, altered)

    def test_saved_validation_threshold_governs_test_classification(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = evaluate_frozen_test_run(
                self.manifest, self.scores,
                {"segment": 0.95, "track": 0.95},
                model="Example", variant="baseline", run_id="run42",
                output_dir=directory,
            )
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["eer"], 0.0)
            self.assertEqual(rows[0]["fake_miss_rate"], 1.0)
            self.assertEqual(rows[0]["hter"], 0.5)
            self.assertTrue((Path(directory) / "example_baseline_test_track_scores.csv").exists())


if __name__ == "__main__":
    unittest.main()
