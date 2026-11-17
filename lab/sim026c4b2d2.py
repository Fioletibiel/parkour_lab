"""Source-free synthetic connected-joint MuJoCo 3.3.6 mechanics benchmark.

Public-only toy masses/geometry. No original anatomical model or source data.
This is NOT a knee, not an orthosis, and has no clinical interpretation.
"""
from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET

import numpy as np

VERSION = "3.3.6"
ANGLES = (0.0, 0.1, 0.3, 0.6, 0.9, 1.2, 1.6)
COEFF = (0.0, 0.03, 0.02, 0.0, 0.0)
STEP_SECONDS = 0.001
STEP_COUNT = 500
TORQUE_NM = 0.1
MODEL_XML = """<mujoco model="public_synthetic_coupling">
  <compiler angle="radian"/>
  <option timestep="0.001" gravity="0 0 0" integrator="implicitfast"/>
  <worldbody>
    <body name="proximal">
      <joint name="q" type="hinge" axis="0 0 1" damping="0.2"/>
      <inertial pos="0.1 0 0" mass="1.1" diaginertia="0.04 0.04 0.02"/>
      <body name="distal" pos="1 0 0">
        <joint name="s" type="slide" axis="1 0 0" damping="0.2"/>
        <inertial pos="0.2 0 0" mass="0.6" diaginertia="0.01 0.02 0.02"/>
      </body>
    </body>
  </worldbody>
  <equality>
    <joint name="coupling" joint1="s" joint2="q"
           polycoef="0 .03 .02 0 0" solref="0.005 1" solimp="0.99 0.99 0.001"/>
  </equality>
</mujoco>"""


def dependent_coordinate(q: float) -> float:
    """The exact synthetic equation, with q in radians and s in meters."""
    q = float(q)
    if not math.isfinite(q):
        raise ValueError("non-finite primary coordinate")
    return 0.03 * q + 0.02 * q * q


def guard_xml(xml: str) -> None:
    """Fail closed on mutations to the published synthetic fixture."""
    root = ET.fromstring(xml)
    if root.tag != "mujoco" or root.get("model") != "public_synthetic_coupling":
        raise ValueError("unexpected public test fixture")
    world = root.find("worldbody")
    if world is None:
        raise ValueError("no world")
    parents = world.findall("body")
    if len(parents) != 1 or parents[0].get("name") != "proximal":
        raise ValueError("not a single root body")
    children = parents[0].findall("body")
    if len(children) != 1 or children[0].get("name") != "distal":
        raise ValueError("not a connected child")
    if children[0].get("pos") != "1 0 0":
        raise ValueError("mutated child anchor")
    joints = (parents[0].findall("joint"), children[0].findall("joint"))
    for item, (name, kind, axis) in zip(joints, (("q", "hinge", "0 0 1"), ("s", "slide", "1 0 0"))):
        if len(item) != 1 or (item[0].get("name"), item[0].get("type"), item[0].get("axis")) != (name, kind, axis):
            raise ValueError("mutated native joint")
    for body in (parents[0], children[0]):
        if body.find("inertial") is None or body.findall("geom") or body.findall("freejoint"):
            raise ValueError("unsupported geometry or mass")
    eqroot = root.find("equality")
    eqs = [] if eqroot is None else eqroot.findall("joint")
    if len(eqs) != 1 or (eqs[0].get("joint1"), eqs[0].get("joint2")) != ("s", "q"):
        raise ValueError("mutated native equality direction")
    coeff = tuple(float(v) for v in eqs[0].get("polycoef", "").split())
    if len(coeff) != 5 or not all(math.isfinite(x) for x in coeff) or coeff != COEFF:
        raise ValueError("mutated native equation coefficients")


def compile_native(xml: str = MODEL_XML):
    import mujoco
    if mujoco.__version__ != VERSION:
        raise ValueError("native engine version mismatch")
    guard_xml(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    if (model.nbody, model.njnt, model.nq, model.nv, model.neq, model.ngeom) != (3, 2, 2, 2, 1, 0):
        raise ValueError("wrong source-free mechanism topology")
    if not np.allclose(model.qpos0, (0, 0), atol=1e-14):
        raise ValueError("unexpected equality reference qpos0")
    if tuple(model.eq_data[0][:5]) != COEFF:
        raise ValueError("native compiled coupling changed")
    if not np.allclose(model.body_mass[1:], (1.1, 0.6), atol=1e-12):
        raise ValueError("modified public inertias")
    qid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "q")
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "s")
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "distal")
    if (model.jnt_qposadr[qid], model.jnt_qposadr[sid]) != (0, 1) or bid != 2:
        raise ValueError("unexpected native joint coordinate order")
    return mujoco, model, bid


def expected_body(q: float):
    p = 1.0 + dependent_coordinate(q)
    return np.asarray((p * math.cos(q), p * math.sin(q), 0.0)), np.asarray((
        (math.cos(q), -math.sin(q), 0.0),
        (math.sin(q), math.cos(q), 0.0),
        (0.0, 0.0, 1.0),
    ))


def pose_checks(mu, model, body_id):
    data = mu.MjData(model)
    max_position = max_rotation = max_constraint = 0.0
    for q in ANGLES:
        data.qpos[:] = (q, dependent_coordinate(q))
        data.qvel[:] = 0.0
        mu.mj_forward(model, data)
        pos, mat = expected_body(q)
        error = float(np.linalg.norm(data.xpos[body_id] - pos))
        # SO(3) geodesic robust for tiny floating point rotations.
        estimated = data.xmat[body_id].reshape(3, 3)
        relative = mat.T @ estimated
        off_diag = np.array((relative[2, 1] - relative[1, 2],
                             relative[0, 2] - relative[2, 0],
                             relative[1, 0] - relative[0, 1]))
        angle = math.atan2(float(np.linalg.norm(off_diag)) / 2.0,
                           float(np.trace(relative) - 1) / 2.0)
        constraint = float(np.max(np.abs(data.efc_pos))) if data.nefc else math.inf
        max_position = max(max_position, error)
        max_rotation = max(max_rotation, angle)
        max_constraint = max(max_constraint, constraint)
        if error > 1e-10 or angle > 1e-10 or constraint > 1e-8:
            raise AssertionError("native analytic FK or equality residual failed")
    return {"states": len(ANGLES), "max_position_error_m": max_position,
            "max_rotation_error_rad": max_rotation,
            "max_initial_constraint_m": max_constraint}


def dynamic_checks(mu, model, applied_torque):
    if applied_torque not in (0.0, TORQUE_NM):
        raise ValueError("out-of-protocol applied torque")
    data = mu.MjData(model)
    if abs(model.opt.timestep - STEP_SECONDS) > 1e-15 or np.linalg.norm(model.opt.gravity) > 1e-15:
        raise ValueError("not the pinned integration environment")
    mu.mj_forward(model, data)
    peak_residual = peak_speed = 0.0
    for k in range(STEP_COUNT):
        data.qfrc_applied[:] = (applied_torque, 0.0)
        mu.mj_step(model, data)
        if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))
                and np.all(np.isfinite(data.xpos))):
            raise AssertionError("non-finite native solver trajectory")
        if any(w.number for w in data.warning):
            raise AssertionError("native engine warning")
        q, s = map(float, data.qpos)
        error = abs(s - dependent_coordinate(q))
        if k >= 25:
            peak_residual = max(peak_residual, error)
            if error > 0.005:
                raise AssertionError("native constraint diverged")
        peak_speed = max(peak_speed, float(np.max(np.abs(data.qvel))))
        if peak_speed >= 25.0:
            raise AssertionError("unphysical toy velocity")
    result = {"steps": STEP_COUNT, "q_final_rad": float(data.qpos[0]),
              "max_constraint_residual_m": peak_residual, "max_speed": peak_speed}
    if applied_torque == 0 and max(np.max(np.abs(data.qpos)), peak_speed) > 1e-10:
        raise AssertionError("zero-gravity resting native joint drift")
    if applied_torque != 0 and not 0.005 < data.qpos[0] < 1.7:
        raise AssertionError("native joint failed frozen response threshold")
    return result


def non_global_quartic_counterexample():
    """A completely synthetic C² spline-like piecewise cubic, no source data."""
    train = np.asarray((0.0, 0.25, 0.50, 0.75, 1.0))
    heldout = np.asarray((0.125, 0.375, 0.625, 0.875, 1.125, 1.375))
    f = lambda x: 0.03*x*x + 0.02*np.maximum(x-0.7, 0.0)**3
    matrix = np.vander(train, N=5, increasing=True)
    coeff = np.linalg.solve(matrix, f(train))
    error = float(np.max(np.abs(np.polynomial.polynomial.polyval(heldout, coeff)-f(heldout))))
    if error <= 1e-8:
        raise AssertionError("global quartic falsely represents non-quartic toy")
    return {"train": len(train), "heldout": len(heldout), "heldout_error_m": error,
            "original_opensim_source_knee_accuracy": "NOT_TESTED"}


def main():
    mu, model, child = compile_native()
    results = {
        "public_synthetic_fixture": "independent connected two-body hinge-slide",
        "engine": mu.__version__,
        "native_fk": pose_checks(mu, model, child),
        "native_idle": dynamic_checks(mu, model, 0.0),
        "native_torque": dynamic_checks(mu, model, TORQUE_NM),
        "nonquartic_toy": non_global_quartic_counterexample(),
        "private_source_inputs": "NONE",
        "full_source_anatomical_mapping": "NOT_TESTED",
        "real_human_orthosis": "NO_GO",
    }
    print(json.dumps(results, sort_keys=True))
    return results


if __name__ == "__main__":
    main()
