"""Public-only synthetic native MuJoCo 3.3.6 controls: 23 assertions."""
import math
import unittest
import numpy as np
from lab import sim026c4b2d2 as case


class ConnectedToyNative(unittest.TestCase):
    def test_01_engine_and_topology(self):
        mu, m, child = case.compile_native()
        self.assertEqual(mu.__version__, "3.3.6")
        self.assertEqual((m.nbody, m.njnt, m.nq, m.nv, m.neq, m.ngeom), (3,2,2,2,1,0))
        self.assertEqual(child, 2)

    def test_02_native_forward_kinematics(self):
        mu, m, child = case.compile_native()
        self.assertEqual(case.pose_checks(mu, m, child)["states"], 7)

    def test_03_native_idle_dynamics(self):
        mu, m, _ = case.compile_native()
        self.assertEqual(case.dynamic_checks(mu, m, 0.)["steps"], 500)

    def test_04_native_driven_dynamics(self):
        mu, m, _ = case.compile_native()
        self.assertGreater(case.dynamic_checks(mu, m, 0.1)["q_final_rad"], 0.005)

    def test_05_piecewise_cubic_not_global_quartic(self):
        self.assertGreater(case.non_global_quartic_counterexample()["heldout_error_m"], 1e-8)

    def test_06_exact_zero_coordinate(self):
        self.assertEqual(case.dependent_coordinate(0), 0.)

    def test_07_quadratic_coordinate(self):
        self.assertAlmostEqual(case.dependent_coordinate(1), .05)

    def test_08_reject_nan(self):
        with self.assertRaises(ValueError):
            case.dependent_coordinate(math.nan)

    def test_09_reject_inf(self):
        with self.assertRaises(ValueError):
            case.dependent_coordinate(math.inf)

    def test_10_zero_pose(self):
        p, r = case.expected_body(0)
        np.testing.assert_allclose(p, (1,0,0), atol=0)
        np.testing.assert_allclose(r, np.eye(3), atol=0)

    def test_11_connected_child_anchor_mutation(self):
        with self.assertRaises(ValueError):
            case.guard_xml(case.MODEL_XML.replace('name="distal" pos="1 0 0"','name="distal" pos="-1 0 0"'))

    def test_12_missing_equality_mutation(self):
        with self.assertRaises(ValueError):
            case.guard_xml(case.MODEL_XML.replace("<equality>","<not_equality>").replace("</equality>","</not_equality>"))

    def test_13_reversed_equality(self):
        with self.assertRaises(ValueError):
            case.guard_xml(case.MODEL_XML.replace('joint1="s" joint2="q"','joint1="q" joint2="s"'))

    def test_14_missing_equality_polynomial(self):
        with self.assertRaises(ValueError):
            case.guard_xml(case.MODEL_XML.replace('polycoef="0 .03 .02 0 0"','polycoef="0 0 0 0 0"'))

    def test_15_negative_coefficient(self):
        with self.assertRaises(ValueError):
            case.guard_xml(case.MODEL_XML.replace('polycoef="0 .03 .02 0 0"','polycoef="0 -.03 .02 0 0"'))

    def test_16_nonfinite_poly(self):
        with self.assertRaises(ValueError):
            case.guard_xml(case.MODEL_XML.replace('polycoef="0 .03 .02 0 0"','polycoef="0 nan .02 0 0"'))

    def test_17_mirror_primary_joint_axis(self):
        with self.assertRaises(ValueError):
            case.guard_xml(case.MODEL_XML.replace('axis="0 0 1"','axis="0 0 -1"'))

    def test_18_mirror_secondary_joint_axis(self):
        with self.assertRaises(ValueError):
            case.guard_xml(case.MODEL_XML.replace('axis="1 0 0"','axis="-1 0 0"'))

    def test_19_wrong_joint_kind(self):
        with self.assertRaises(ValueError):
            case.guard_xml(case.MODEL_XML.replace('name="q" type="hinge"','name="q" type="slide"'))

    def test_20_wrong_child_name(self):
        with self.assertRaises(ValueError):
            case.guard_xml(case.MODEL_XML.replace('name="distal" pos="1 0 0"','name="distal_other" pos="1 0 0"'))

    def test_21_missing_mass(self):
        with self.assertRaises(ValueError):
            case.guard_xml(case.MODEL_XML.replace('<inertial pos="0.2 0 0"','<notinertial pos="0.2 0 0"'))

    def test_22_negative_mass_native_failure(self):
        with self.assertRaises(Exception):
            case.compile_native(case.MODEL_XML.replace('mass="1.1"','mass="-1.1"'))

    def test_23_missing_gravity_free_load(self):
        with self.assertRaises(ValueError):
            case.dynamic_checks(*case.compile_native()[:2], .2)


if __name__ == "__main__":
    unittest.main()
