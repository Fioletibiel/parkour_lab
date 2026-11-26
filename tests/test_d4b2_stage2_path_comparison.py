"""Source-free, paired-comparison P1b regression tests."""
import math
from pathlib import Path
import unittest

from lab import d4b2_stage2_path_comparison as fit

KEYS=("path_length_rms_mm_overall","path_length_rms_mm_median_muscle",
      "path_length_rms_mm_p95_muscle","path_length_rms_mm_worst_muscle",
      "path_length_abs_max_mm","moment_arm_rms_mm_mean_scored_pair",
      "moment_arm_rms_mm_median_scored_pair",
      "moment_arm_rms_mm_worst_scored_pair",
      "moment_arm_rms_pct_source_peak_mean")


def sample(v=10):
    o={k:float(v) for k in KEYS}
    o.update({"classification":"SCREEN_VALID_MEASURED","cohort":"core",
              "source_native_lengths_compared":320,
              "scored_pairs_above_1mm_source_peak":76})
    return o


class Stage2ComparisonTest(unittest.TestCase):
    def test_explicit_error_reduction(self):
        r=fit.compare(sample(20),sample(5))
        self.assertTrue(r["max_path_length_error_improved"])
        self.assertTrue(r["max_scored_moment_arm_rms_improved"])
        self.assertEqual(r["error_change_mm_or_pct_stage2_minus_stage1"]["path_length_abs_max_mm"],-15)
        self.assertEqual(r["muscle_accuracy_pass"],"NOT_ESTABLISHED")

    def test_worsening_not_mislabeled_as_improvement(self):
        r=fit.compare(sample(5),sample(20))
        self.assertFalse(r["max_path_length_error_improved"])
        self.assertFalse(r["max_scored_moment_arm_rms_improved"])

    def test_unchanged_not_mislabeled_as_improvement(self):
        r=fit.compare(sample(5),sample(5))
        self.assertFalse(r["max_path_length_error_improved"])
        self.assertFalse(r["max_scored_moment_arm_rms_improved"])

    def test_invalid_cohort_rejected(self):
        bad=sample(10)
        bad["classification"]="SCREEN_INVALID_FD"
        with self.assertRaises(ValueError):
            fit.compare(bad,sample(5))

    def test_coverage_mismatch_rejected(self):
        bad=sample(5)
        bad["source_native_lengths_compared"]=160
        with self.assertRaises(ValueError):
            fit.compare(sample(20),bad)

    def test_pre_registered_2023_model_only(self):
        self.assertEqual(fit.MODEL_NAME,"RajagopalLaiUhlrich2023")
        self.assertAlmostEqual(fit.BASELINE_STAGE1_CORE_WORST_LENGTH_MM,203.95778123)
        self.assertAlmostEqual(fit.BASELINE_STAGE1_EXTRA_WORST_LENGTH_MM,9.44103028)
        self.assertEqual(fit.BASELINE_REPEAT_TOL_MM,.01)

    def test_public_protocol_preserves_stage1_stage2_comparison(self):
        p=(Path(__file__).resolve().parents[1]/"lab"/
           "D4B2_P1B_STAGE2_PROTOCOL_2026-11-26.md").read_text()
        self.assertIn("same 4 predeclared core poses and separately reported 2 high-flexion poses",p.lower())
        self.assertIn("no absolute path accuracy PASS",p)
        self.assertIn("no muscles removed",p)

if __name__=="__main__":
    unittest.main()
