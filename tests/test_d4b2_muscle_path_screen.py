"""D4B2-P1a source-free validity, coverage and mutation checks."""
from __future__ import annotations

import math
import unittest

import numpy as np
from lab import d4b2_muscle_path_screen as p1


def fixture(pose_count=4):
    os_len=np.full((pose_count,80),.25)
    mj_len=np.full((pose_count,80),.2502)
    src_ma=np.full((pose_count,80,4),.002)
    mj_ma=np.full((pose_count,80,4),.0022)
    return (os_len,mj_len,src_ma,mj_ma,src_ma.copy(),mj_ma.copy())


class PublicMusclePathContract(unittest.TestCase):
    def test_nominal_full_coverage(self):
        r=p1.summarize_errors(*fixture(),cohort="core")
        self.assertEqual(r["classification"],"SCREEN_VALID_MEASURED")
        self.assertEqual(r["source_native_lengths_compared"],320)
        self.assertEqual(r["coordinate_derivative_samples"],1280)
        self.assertEqual(r["scored_pairs_above_1mm_source_peak"],320)
        self.assertAlmostEqual(r["path_length_rms_mm_overall"],.2)
        self.assertEqual(r["absolute_muscle_accuracy_gate"],"NOT_PREDECLARED")

    def test_2023_extended_coverage(self):
        r=p1.summarize_errors(*fixture(2),cohort="2023_only")
        self.assertEqual(r["source_native_lengths_compared"],160)
        self.assertEqual(r["coordinate_derivative_samples"],640)

    def test_nonfinite_values_are_fatal(self):
        for k in range(6):
            args=list(fixture())
            args[k][0,0]=math.nan
            with self.subTest(field=k),self.assertRaises(ValueError):
                p1.summarize_errors(*args,cohort="core")

    def test_missing_muscles_or_missing_derivative_dimension_are_fatal(self):
        args=list(fixture())
        args[0]=args[0][:,:79]
        with self.assertRaises(ValueError):
            p1.summarize_errors(*args,cohort="core")
        args=list(fixture())
        args[2]=args[2][:,:,:3]
        with self.assertRaises(ValueError):
            p1.summarize_errors(*args,cohort="core")

    def test_near_zero_reference_pairs_not_scored(self):
        args=list(fixture())
        args[2][:,:,:]=.0002
        args[4][:,:,:]=.0002
        with self.assertRaises(ValueError):
            p1.summarize_errors(*args,cohort="core")

    def test_fd_inconsistent_is_not_pass(self):
        args=list(fixture())
        args[4][:,:,:]+=2e-6
        r=p1.summarize_errors(*args,cohort="core")
        self.assertEqual(r["classification"],"SCREEN_INVALID_FD")
        self.assertLess(r["source_fd_h_vs_10h_consistency_fraction"],.99)

    def test_large_muscle_error_does_not_fake_accuracy_pass(self):
        args=list(fixture())
        args[1][:,:]=1.20
        r=p1.summarize_errors(*args,cohort="core")
        self.assertEqual(r["classification"],"SCREEN_VALID_MEASURED")
        self.assertGreater(r["path_length_rms_mm_overall"],900.)
        self.assertEqual(r["absolute_muscle_accuracy_gate"],"NOT_PREDECLARED")

    def test_nonfinite_fraction_checker(self):
        for a in (np.array([]),np.array([np.inf]),np.array([np.nan])):
            with self.assertRaises(ValueError):
                p1.safe_fraction_good(a)

    def test_protocol_frozen_and_public_only(self):
        from pathlib import Path
        root=Path(__file__).resolve().parents[1]
        protocol=(root/"lab"/"D4B2_MUSCLE_PATH_SCREEN_PROTOCOL_2026-11-26.md").read_text()
        self.assertIn("No stage 2 muscle-path fitting",protocol)
        self.assertIn("No stage 2 muscle-path fitting",protocol)
        self.assertIn("99%",protocol)
        self.assertIn("all **80 named source muscles**".lower(),protocol.lower())
        self.assertEqual(p1.N_MUSCLES,80)
        self.assertEqual(p1.FD_H,1e-6)
        self.assertEqual(p1.FD_CHECK_H,1e-5)


if __name__=="__main__":
    unittest.main()
