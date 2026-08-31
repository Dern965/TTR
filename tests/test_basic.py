from __future__ import annotations

import unittest

from src.inference.predictor import EyePrediction, combine_eye_predictions
from src.inference.temporal_detector import PredictionSmoother, TemporalClosureDetector


class TemporalClosureDetectorTests(unittest.TestCase):
    def test_short_closure_does_not_alert(self) -> None:
        detector = TemporalClosureDetector(threshold_seconds=1.5)
        first = detector.update("CLOSED", now=10.0)
        second = detector.update("CLOSED", now=10.6)
        self.assertFalse(first.alert)
        self.assertFalse(second.alert)
        self.assertAlmostEqual(second.closed_duration, 0.6, places=6)

    def test_sustained_closure_alerts(self) -> None:
        detector = TemporalClosureDetector(threshold_seconds=1.5)
        detector.update("CLOSED", now=5.0)
        status = detector.update("CLOSED", now=6.6)
        self.assertTrue(status.alert)
        self.assertGreaterEqual(status.closed_duration, 1.5)

    def test_open_resets_timer(self) -> None:
        detector = TemporalClosureDetector(threshold_seconds=1.0)
        detector.update("CLOSED", now=2.0)
        detector.update("OPEN", now=2.8)
        status = detector.update("CLOSED", now=3.0)
        self.assertFalse(status.alert)
        self.assertEqual(status.closed_duration, 0.0)

    def test_short_uncertain_gap_preserves_closure(self) -> None:
        detector = TemporalClosureDetector(
            threshold_seconds=1.0,
            uncertain_grace_period=0.3,
        )
        detector.update("CLOSED", now=1.0)
        gap = detector.update("UNCERTAIN", now=1.15)
        resumed = detector.update("CLOSED", now=1.4)
        self.assertGreater(gap.closed_duration, 0.0)
        self.assertAlmostEqual(resumed.closed_duration, 0.4, places=6)

    def test_long_uncertain_gap_resets_closure(self) -> None:
        detector = TemporalClosureDetector(
            threshold_seconds=1.0,
            uncertain_grace_period=0.2,
        )
        detector.update("CLOSED", now=1.0)
        detector.update("UNCERTAIN", now=1.1)
        after_grace = detector.update("UNCERTAIN", now=1.4)
        self.assertEqual(after_grace.closed_duration, 0.0)
        self.assertFalse(after_grace.alert)


class PredictionSmootherTests(unittest.TestCase):
    def test_majority_vote(self) -> None:
        smoother = PredictionSmoother(history_size=5, max_uncertain_frames=2)
        states = ["OPEN", "OPEN", "CLOSED", "OPEN", "CLOSED"]
        result = "UNCERTAIN"
        for state in states:
            result = smoother.update(state)
        self.assertEqual(result, "OPEN")

    def test_short_uncertain_is_tolerated(self) -> None:
        smoother = PredictionSmoother(history_size=3, max_uncertain_frames=2)
        smoother.update("CLOSED")
        result = smoother.update("UNCERTAIN")
        self.assertEqual(result, "CLOSED")

    def test_too_many_uncertain_frames_clear_history(self) -> None:
        smoother = PredictionSmoother(history_size=3, max_uncertain_frames=1)
        smoother.update("CLOSED")
        smoother.update("UNCERTAIN")
        result = smoother.update("UNCERTAIN")
        self.assertEqual(result, "UNCERTAIN")


class CombineEyePredictionTests(unittest.TestCase):
    def test_both_closed(self) -> None:
        left = EyePrediction("CLOSED", 0.95, 0.05)
        right = EyePrediction("CLOSED", 0.91, 0.09)
        result = combine_eye_predictions(left, right)
        self.assertEqual(result.state, "CLOSED")
        self.assertEqual(result.source, "BOTH_EYES")

    def test_disagreement_is_uncertain(self) -> None:
        left = EyePrediction("CLOSED", 0.95, 0.05)
        right = EyePrediction("OPEN", 0.93, 0.93)
        result = combine_eye_predictions(left, right)
        self.assertEqual(result.state, "UNCERTAIN")

    def test_one_strong_eye_can_be_used(self) -> None:
        left = EyePrediction("CLOSED", 0.96, 0.04)
        result = combine_eye_predictions(
            left,
            None,
            allow_single_eye_fallback=True,
            single_eye_min_confidence=0.85,
        )
        self.assertEqual(result.state, "CLOSED")
        self.assertEqual(result.source, "SINGLE_EYE")

    def test_one_weak_eye_is_uncertain(self) -> None:
        left = EyePrediction("CLOSED", 0.72, 0.28)
        result = combine_eye_predictions(
            left,
            None,
            allow_single_eye_fallback=True,
            single_eye_min_confidence=0.85,
        )
        self.assertEqual(result.state, "UNCERTAIN")


if __name__ == "__main__":
    unittest.main()
