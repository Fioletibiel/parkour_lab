"""Source-free D3B independent Lagrange, energy and native CI mutation tests."""
import math
import unittest
import numpy as np
from lab import sim026c4b2d3a as a
from lab import sim026c4b2d3b as b


class AnalyticalIndependentReference(unittest.TestCase):
    def test_inertial_mass_at_zero_is_positive(self):
        self.assertGreater(b.energy_mass(0),.5)

    def test_reduced_mass_matches_kinetic_energy(self):
        for q in (.4,.69,.7,.71,1.2):
            v=.63
            s=b.f(q)
            vs=b.f(q,1)*v
            e=b.native_energy(q,s,v,vs,0)
            self.assertAlmostEqual(e,.5*b.energy_mass(q)*v*v,places=12)

    def test_mass_derivative_by_independent_finite_difference(self):
        for q in (.45,.65,.75,1.15):
            h=1e-6
            fd=(b.energy_mass(q+h)-b.energy_mass(q-h))/(2*h)
            self.assertLess(abs(fd-b.energy_mass_derivative(q)),1e-8)

    def test_gravity_potential_gradient(self):
        for q in (.4,.66,.75,1.2):
            h=1e-6
            fd=(b.potential(q+h,1)-b.potential(q-h,1))/(2*h)
            self.assertLess(abs(fd-b.potential_gradient(q,1)),1e-8)

    def test_potential_zero_gravity(self):
        self.assertEqual(b.potential(.68,0),0)
        self.assertEqual(b.potential_gradient(.68,0),0)

    def test_viscous_power_dissipates(self):
        self.assertLess(b.native_power(.8,.1,0),0)

    def test_lagrange_zero_velocity_zero_gravity_no_effort(self):
        self.assertEqual(b.reduced_acceleration(.68,0,0,0),0)

    def test_positive_effort_initial_acceleration(self):
        self.assertGreater(b.reduced_acceleration(.68,0,.1,0),0)

    def test_negative_effort_initial_acceleration(self):
        self.assertLess(b.reduced_acceleration(.72,0,-.1,0),0)

    def test_torque_time_boundary(self):
        self.assertEqual(b.torque_at("PULSE",.07),.12)
        self.assertEqual(b.torque_at("PULSE",.08),-.04)

    def test_reject_unknown_torque_scenario(self):
        with self.assertRaises(ValueError):
            b.torque_at("PATIENT",0)

    def test_reject_negative_original_curve_coordinate(self):
        with self.assertRaises(ValueError):
            b.f(-.01)

    def test_reject_nonfinite_curve_coordinate(self):
        with self.assertRaises(ValueError):
            b.f(math.nan)

    def test_reject_wrong_step_size(self):
        with self.assertRaises(ValueError):
            b.run_case("FORWARD",.68,.8,0,.003)

    def test_native_compiled_source_free_mass_and_coefficients(self):
        mu,m=a.compiled()
        np.testing.assert_allclose(m.body_mass[1:],(1.1,.6),atol=1e-12)
        self.assertEqual(tuple(m.eq_data[0,:5]),a.LOW)
        self.assertEqual(tuple(m.eq_data[1,:5]),a.HIGH)

    def test_mutation_invalid_spline_inequality_rejected(self):
        with self.assertRaises(ValueError):
            a.verify_xml_fixture(a.XML.replace('polycoef="-.00686 .0594 -.022 .02 0"',
                                                'polycoef="-.00686 .0594 -.021 .02 0"'))

    def test_mutation_both_native_constraints_active_rejected(self):
        with self.assertRaises(ValueError):
            a.verify_xml_fixture(a.XML.replace('active="false"','active="true"'))

    def test_mutation_wrong_native_axis_rejected(self):
        with self.assertRaises(ValueError):
            a.verify_xml_fixture(a.XML.replace('axis="0 0 1"','axis="0 1 0"'))

    def test_reference_RK4_tracks_physical_forward_time(self):
        y=np.array((.68,.8))
        z=b.reference_step(y,0,.001,"FORWARD",0)
        self.assertGreater(z[0],y[0])
        self.assertLess(z[1],1.)

    def test_reference_RK4_reverse_direction(self):
        y=np.array((.72,-.8))
        z=b.reference_step(y,0,.001,"REVERSE",0)
        self.assertLess(z[0],y[0])

    def test_native_forward_crossing(self):
        out=b.run_case("FORWARD",.68,.8,0,.001)
        self.assertGreaterEqual(out["switch_count"],1)

    def test_native_reverse_crossing(self):
        out=b.run_case("REVERSE",.72,-.8,0,.001)
        self.assertGreaterEqual(out["switch_count"],1)

    def test_native_gravity_crossing(self):
        out=b.run_case("GRAVITY",.68,.8,1.,.001)
        self.assertGreaterEqual(out["switch_count"],1)

    def test_all_preregistered_twelve_native_cases(self):
        out=b.run_all()
        self.assertEqual((out["cases"],out["scenarios"],out["timesteps"]),(12,4,3))
        self.assertTrue(out["sampled_force_jump_does_not_prove_pointwise_continuity"])


if __name__=="__main__":
    unittest.main()
