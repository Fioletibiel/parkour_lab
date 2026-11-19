"""Independent negative controls for original-authored synthetic D4A converter fixture."""
import unittest
from lab import sim026c4b2d4a_converter_pilot as p


class D4AConverterPilotIntegrity(unittest.TestCase):
    def test_01_upstream_pin(self):
        self.assertEqual(len(p.UPSTREAM_COMMIT),40)
        self.assertEqual(p.UPSTREAM_COMMIT,"8c4d97e96f43c0d1676b8d58b0cff95af72a0f0f")

    def test_02_exact_source_free_masses(self):
        self.assertEqual(p.MASS,(12.,4.))
        self.assertEqual(sum(p.MASS),16.)

    def test_03_connected_topology_labels(self):
        self.assertEqual(p.NAMES,("proximal","distal"))
        self.assertEqual(p.DOFS,("q_hip","q_knee"))

    def test_04_native_pose_heldouts(self):
        self.assertEqual(p.POSITIONS,((0.,0.),(.3,-.2),(.9,-.7)))

    def test_05_different_source_and_native_versions_not_conflated(self):
        self.assertIn("3.15.0",p.check_compiled.__code__.co_consts)
        self.assertNotIn("3.3.6",p.check_compiled.__code__.co_consts)

    def test_06_no_source_original_assets_embedded(self):
        self.assertNotIn("Rajagopal",p.build_model.__doc__ or "")
        self.assertNotIn("patient",str(p.MASS))

    def test_07_reject_unknown_native_converted_file(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(AssertionError):
                p.discover_converted(folder)
            (Path(folder)/"a.xml").write_text("<mujoco/>")
            (Path(folder)/"b.xml").write_text("<mujoco/>")
            with self.assertRaises(AssertionError):
                p.discover_converted(folder)

    def test_08_positive_three_native_fk_finite_preconditions(self):
        import math
        for q0,q1 in p.POSITIONS:
            self.assertTrue(math.isfinite(q0) and math.isfinite(q1))


if __name__=="__main__":
    unittest.main()
