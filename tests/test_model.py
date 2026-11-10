"""
Unit tests for Student Performance & At-Risk Prediction model pipeline.
Run with: python -m unittest discover tests
"""

import unittest
from pathlib import Path
import joblib
import pandas as pd
import numpy as np


class TestModelArtifact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Locate project root and model path
        cls.root_dir = Path(__file__).resolve().parent.parent
        cls.model_path = cls.root_dir / "models" / "student_risk_model.joblib"
        cls.dataset_path = cls.root_dir / "data" / "processed" / "student_performance_synthetic.csv"

        cls.assertTrue(cls.model_path.exists(), f"Model artifact not found at {cls.model_path}")
        cls.bundle = joblib.load(cls.model_path)

    def test_bundle_structure(self):
        """Verify that the model bundle contains all expected keys."""
        required_keys = ["model", "threshold", "features"]
        for key in required_keys:
            self.assertIn(key, self.bundle, f"Bundle missing key: {key}")

    def test_threshold_value(self):
        """Verify the decision threshold is valid and tuned to 0.35."""
        threshold = self.bundle["threshold"]
        self.assertIsInstance(threshold, (float, int))
        self.assertGreater(threshold, 0.0)
        self.assertLess(threshold, 1.0)
        self.assertAlmostEqual(threshold, 0.35, places=2)

    def test_features_list(self):
        """Verify that expected input features are present and non-empty."""
        features = self.bundle["features"]
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 6)
        expected = [
            "attendance_percentage",
            "assignment_completion",
            "previous_score",
            "study_hours",
            "participation_score",
            "late_submissions"
        ]
        self.assertEqual(sorted(features), sorted(expected))

    def test_sample_inference(self):
        """Test inference on synthetic high-risk and low-risk student samples."""
        model = self.bundle["model"]
        features = self.bundle["features"]

        # High-risk profile (low attendance, low scores, high late submissions)
        high_risk_student = pd.DataFrame([{
            "attendance_percentage": 45,
            "assignment_completion": 40,
            "previous_score": 35,
            "study_hours": 1.0,
            "participation_score": 30,
            "late_submissions": 6
        }])[features]

        prob_high = model.predict_proba(high_risk_student)[0][1]
        self.assertGreaterEqual(prob_high, 0.0)
        self.assertLessEqual(prob_high, 1.0)
        self.assertGreater(prob_high, self.bundle["threshold"], "Struggling student should exceed risk threshold")

        # Low-risk profile (high attendance, high scores, zero late submissions)
        low_risk_student = pd.DataFrame([{
            "attendance_percentage": 95,
            "assignment_completion": 95,
            "previous_score": 90,
            "study_hours": 6.0,
            "participation_score": 90,
            "late_submissions": 0
        }])[features]

        prob_low = model.predict_proba(low_risk_student)[0][1]
        self.assertLess(prob_low, self.bundle["threshold"], "Strong student should be below risk threshold")

    def test_dataset_reproducibility(self):
        """Verify dataset exists, is non-empty, and has expected class balance."""
        self.assertTrue(self.dataset_path.exists(), "Synthetic dataset file does not exist")
        df = pd.read_csv(self.dataset_path)
        self.assertEqual(len(df), 5000, "Dataset should contain exactly 5,000 records")
        self.assertIn("at_risk", df.columns, "Target column 'at_risk' missing from dataset")
        risk_rate = df["at_risk"].mean()
        self.assertAlmostEqual(risk_rate, 0.151, delta=0.01, msg="Expected ~15.1% at-risk class balance")


if __name__ == "__main__":
    unittest.main()

