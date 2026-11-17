"""D3A wholly synthetic switched two-joint MuJoCo equality tests."""
import math
import unittest

import numpy as np

from lab import sim026c4b2d3a as m


class PiecewiseNative(unittest.TestCase):
    def test_continuity_C2_and_third_derivative_jump(self):
        self.assertAlmostEqual(m.check_continuity()["third_derivative_jump"],.12)

    def test_global_quartic_fails_holdout(self):
        self.assertGreater(m.global_quartic_failure()["max_global_quartic_holdout_error_m"],1e-7)

    def test_compiled_native_topology(self):
        mu,model=m.compiled()
        self.assertEqual((model.nbody,model.njnt,model.nq,model.neq,model.ngeom),(3,2,2,2,0))
        self.assertEqual(mu.__version__,"3.3.6")

    def test_native_static_two_polynomial_branches(self):
        mu,model=m.compiled()
        self.assertEqual(m.static_native(mu,model)["poses"],11)

    def test_native_idle_500_steps(self):
        mu,model=m.compiled()
        self.assertEqual(m.native_dynamics(mu,model,False)["steps"],500)

    def test_native_dynamic_knot_crossing_500_steps(self):
        mu,model=m.compiled()
        outcome=m.native_dynamics(mu,model,True)
        self.assertGreaterEqual(outcome["switches"],1)

    def test_knot_low_is_one_hot(self):
        mu,model=m.compiled()
        d=mu.MjData(model)
        self.assertEqual(m.select_active(d,.7),0)
        self.assertEqual(list(d.eq_active),[1,0])

    def test_above_knot_high_is_one_hot(self):
        mu,model=m.compiled()
        d=mu.MjData(model)
        self.assertEqual(m.select_active(d,.700001),1)
        self.assertEqual(list(d.eq_active),[0,1])

    def test_wrong_low_coefficient_rejected(self):
        altered=m.XML.replace('polycoef="0 .03 .02 0 0"','polycoef="0 .03 .03 0 0"')
        with self.assertRaises(ValueError):
            m.verify_xml_fixture(altered)

    def test_wrong_high_coefficient_rejected(self):
        altered=m.XML.replace('polycoef="-.00686 .0594 -.022 .02 0"','polycoef="-.00686 .0594 -.02 .02 0"')
        with self.assertRaises(ValueError):
            m.verify_xml_fixture(altered)

    def test_missing_high_equality_rejected(self):
        altered=m.XML.replace('name="high"','name="high_omitted"')
        with self.assertRaises(ValueError):
            m.verify_xml_fixture(altered)

    def test_both_equalities_on_rejected(self):
        altered=m.XML.replace('active="false"','active="true"')
        with self.assertRaises(ValueError):
            m.verify_xml_fixture(altered)

    def test_swapped_constraint_joint_direction_rejected(self):
        altered=m.XML.replace('joint1="s" joint2="q"','joint1="q" joint2="s"')
        with self.assertRaises(ValueError):
            m.verify_xml_fixture(altered)

    def test_flipped_hinge_axis_rejected(self):
        altered=m.XML.replace('axis="0 0 1"','axis="0 0 -1"')
        with self.assertRaises(ValueError):
            m.verify_xml_fixture(altered)

    def test_flipped_slide_axis_rejected(self):
        altered=m.XML.replace('axis="1 0 0"','axis="-1 0 0"')
        with self.assertRaises(ValueError):
            m.verify_xml_fixture(altered)

    def test_renamed_body_rejected(self):
        altered=m.XML.replace('name="child"','name="child_other"')
        with self.assertRaises(ValueError):
            m.verify_xml_fixture(altered)

    def test_missing_inertia_rejected(self):
        altered=m.XML.replace('<inertial pos="0.2 0 0"','<notinertial pos="0.2 0 0"')
        with self.assertRaises(ValueError):
            m.verify_xml_fixture(altered)

    def test_nonfinite_angle_rejected(self):
        with self.assertRaises(ValueError):
            m.target(math.nan)

    def test_out_of_domain_rejected(self):
        with self.assertRaises(ValueError):
            m.target(1.7)

    def test_nonfinite_switch_rejected(self):
        mu,model=m.compiled()
        d=mu.MjData(model)
        with self.assertRaises(ValueError):
            m.select_active(d,math.inf)

    def test_invalid_derivative_rejected(self):
        with self.assertRaises(ValueError):
            m.target(.7,4)

    def test_high_branch_has_nonzero_cubic(self):
        self.assertEqual(m.HIGH[3],.02)

    def test_no_input_derived_from_original_source(self):
        self.assertNotIn("Rajagopal",m.XML)
        self.assertNotIn("<geom",m.XML)
        self.assertEqual(len(m.SAMPLES),11)


if __name__=="__main__":
    unittest.main()
