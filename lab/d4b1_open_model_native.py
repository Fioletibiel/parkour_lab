"""Public-only OpenSim -> MuJoCo source-native reproducibility probe.

All input anatomical models come from a PUBLIC pinned OpenSim distribution.
This script does not access the private Parkour repository, secrets or patients.
Original source, converted XML, geometry and detailed comparison stay ephemeral.
Published output is restricted to aggregate scores and pass/fail classifications.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

SOURCE_COMMIT = "d9b05d470b1a481c222372c85b75772faf8f7792"
CONVERTER_COMMIT = "8c4d97e96f43c0d1676b8d58b0cff95af72a0f0f"
MODELS = (
    ("Rajagopal2016", "f0bbab84ebbd652fece7c36928e916618f0b9fff", 874820),
    ("RajagopalLaiUhlrich2023", "d296dff62a1340a895dfe3a9ae8be03895b2e9f1", 922865),
)
CORE_KNEES = ((0., 0.), (.45, .30), (1.20, 1.10), (1.80, 1.60))
CORE_HIPS = ((0., 0.), (.40, .35), (1.20, 1.10), (1.50, 1.35))
EXTRA_KNEES = ((2.15, 2.20), (2.35, 2.30))
EXTRA_HIPS = ((1.25, 1.10), (1.60, 1.45))
POS_LIMIT_M, ANG_LIMIT_DEG = .001, .5
COM_LIMIT_M, EQ_LIMIT, QERR_LIMIT = .001, 1e-9, 1e-9
SOURCE_ROOT_ENV = "PUBLIC_OPEN_MODEL_SOURCE_ROOT"
CURRENT_STAGE = "not_started"


def stage(value: str) -> None:
    """Whitelisted coarse stage classification, never raw file content."""
    global CURRENT_STAGE
    allowed = {
        "source_identity", "opensim_load", "source_topology", "source_mass",
        "converter_support", "stage1_conversion", "native_compile",
        "core_fk", "2023_extension_fk", "completed",
    }
    if value not in allowed:
        raise ValueError("UNRECOGNIZED_DIAGNOSTIC_STAGE")
    CURRENT_STAGE = value



def check_finite_nonnegative(value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("INVALID_NUMERICAL_MEASUREMENT")
    return result


def model_git_sha1(path: Path) -> str:
    payload = path.read_bytes()
    header = ("blob " + str(len(payload)) + "\0").encode()
    return hashlib.sha1(header + payload).hexdigest()


def converter_command(args: list[str], timeout: int = 1200) -> int:
    # Full stdout/stderr are deliberately not printed. They may reveal original
    # model detail; only the process result and aggregate reports are retained.
    proc = subprocess.run([sys.executable, "-m", "osim2mujoco.cli", *args],
                          capture_output=True, text=True, errors="replace",
                          timeout=timeout, check=False)
    return proc.returncode


def verify_one_cohort(source_model, converted_xml: Path, knees, hips, cohort: str):
    import numpy as np
    import mujoco
    import opensim
    from osim2mujoco.checks import mujoco_compare as resolver

    if not knees or len(knees) != len(hips):
        raise ValueError("INVALID_POSE_SPECIFICATION")
    state = source_model.initSystem()
    coords = source_model.getCoordinateSet()
    coord_names = [coords.get(i).getName() for i in range(coords.getSize())]
    source_defaults = {n: float(coords.get(n).getDefaultValue()) for n in coord_names}
    source_bodies = source_model.getBodySet()
    body_names = [source_bodies.get(i).getName() for i in range(source_bodies.getSize())]
    if len(body_names) != 22:
        raise ValueError("SOURCE_BODY_COUNT_MISMATCH")
    native, _ = resolver.load_without_meshes(str(converted_xml))
    data = mujoco.MjData(native)
    direct, dependent, issues = resolver.joint_plan(native, coord_names)
    if issues:
        raise ValueError("NATIVE_COORDINATE_PLAN_REJECTED")
    ground = mujoco.mj_name2id(native, mujoco.mjtObj.mjOBJ_BODY, "ground")
    ids = {name: mujoco.mj_name2id(native, mujoco.mjtObj.mjOBJ_BODY, name)
           for name in body_names}
    if ground < 0 or any(body_id < 0 for body_id in ids.values()):
        raise ValueError("NATIVE_SOURCE_BODY_MAPPING_INCOMPLETE")
    active = native.eq_active0 if hasattr(native, "eq_active0") else native.eq_active
    n_active = int(np.count_nonzero(active))
    if n_active <= 0:
        raise ValueError("NATIVE_NO_ACTIVE_EQUALITIES")

    couplers = []
    constraints = source_model.getConstraintSet()
    for i in range(constraints.getSize()):
        cc = opensim.CoordinateCouplerConstraint.safeDownCast(constraints.get(i))
        if cc is None:
            raise ValueError("UNSUPPORTED_SOURCE_CONSTRAINT")
        if cc.get_isEnforced():
            indep = cc.getIndependentCoordinateNames()
            couplers.append((cc.getDependentCoordinateName(),
                             [indep.get(j) for j in range(indep.getSize())], cc))

    pos_max = rot_max = com_max = qerr_max = eq_max = 0.
    match_all = True
    for knee, hip in zip(knees, hips):
        values = dict(source_defaults)
        values.update(dict(zip(("knee_angle_r", "knee_angle_l"), knee)))
        values.update(dict(zip(("hip_flexion_r", "hip_flexion_l"), hip)))
        for dependent_name, independents, cc in couplers:
            x = opensim.Vector(len(independents), 0.)
            for j, name in enumerate(independents):
                x.set(j, values[name])
            values[dependent_name] = float(cc.getFunction().calcValue(x))
        for name in coord_names:
            c = coords.get(name)
            if c.getLocked(state):
                if abs(float(c.getValue(state)) - values[name]) > 1e-10:
                    raise ValueError("LOCKED_SOURCE_COORDINATE_CONFLICT")
            else:
                c.setValue(state, values[name], False)
        source_model.realizePosition(state)
        qerr = state.getQErr()
        for i in range(qerr.size()):
            qerr_max = max(qerr_max, check_finite_nonnegative(abs(qerr.get(i))))
        resolver.set_pose(native, data, direct, dependent, values)
        residual, rows = resolver.equality_residual(native, data)
        match_all = match_all and rows == n_active
        eq_max = max(eq_max, check_finite_nonnegative(residual))

        rg = data.xmat[ground].reshape(3, 3)
        pg = data.xpos[ground]
        weighted_com = np.zeros(3)
        total_mass = 0.
        for name in body_names:
            body = source_bodies.get(name)
            transform = body.getTransformInGround(state)
            p = np.array([transform.p().get(i) for i in range(3)])
            rot = np.array([[transform.R().get(i, j) for j in range(3)]
                            for i in range(3)])
            body_id = ids[name]
            actual_pos = rg.T @ (data.xpos[body_id] - pg)
            actual_rot = rg.T @ data.xmat[body_id].reshape(3, 3)
            pos_max = max(pos_max, check_finite_nonnegative(
                float(np.linalg.norm(actual_pos - p))))
            rot_max = max(rot_max, check_finite_nonnegative(
                float(resolver.angle_deg(rot, actual_rot))))
            mass = check_finite_nonnegative(body.getMass())
            weighted_com += mass * (rg.T @ (data.xipos[body_id] - pg))
            total_mass += mass
        if total_mass <= 0:
            raise ValueError("INVALID_SOURCE_BODY_MASS")
        c = source_model.calcMassCenterPosition(state)
        source_com = np.array([c.get(i) for i in range(3)])
        com_max = max(com_max, check_finite_nonnegative(
            float(np.linalg.norm(source_com - weighted_com / total_mass))))

    passes = (match_all and pos_max <= POS_LIMIT_M and rot_max <= ANG_LIMIT_DEG
              and com_max <= COM_LIMIT_M and qerr_max <= QERR_LIMIT
              and eq_max <= EQ_LIMIT)
    return {
        "cohort": cohort, "poses": len(knees), "named_bodies": len(body_names),
        "body_comparisons": len(knees) * len(body_names),
        "native_equality_rows_consistent": match_all,
        "native_active_equalities": n_active,
        "max_body_position_error_mm": round(pos_max * 1000, 8),
        "max_body_orientation_error_deg": round(rot_max, 8),
        "max_source_native_com_error_mm": round(com_max * 1000, 8),
        "max_original_QErr": qerr_max, "max_native_equality_residual": eq_max,
        "position_limit_mm": POS_LIMIT_M * 1000, "angle_limit_deg": ANG_LIMIT_DEG,
        "classification": "PASS" if passes else "FAIL",
    }


def one_model(name: str, expected_blob: str, expected_size: int, source_root: Path, temp: Path):
    import opensim
    import mujoco
    from osim2mujoco.checks import mujoco_compare as resolver

    stage("source_identity")
    path = source_root / "Models" / "Rajagopal" / (name + ".osim")
    if path.stat().st_size != expected_size or model_git_sha1(path) != expected_blob:
        raise ValueError("ORIGINAL_PUBLIC_SOURCE_IDENTITY_MISMATCH")
    stage("opensim_load")
    model = opensim.Model(str(path))
    model.initSystem()
    stage("source_topology")
    bodies = model.getBodySet()
    mass = sum(float(bodies.get(i).getMass()) for i in range(bodies.getSize()))
    if (bodies.getSize(), model.getJointSet().getSize(),
            model.getCoordinateSet().getSize()) != (22, 20, 39):
        raise ValueError("ORIGINAL_SOURCE_TOPOLOGY_MISMATCH")
    stage("source_mass")
    if abs(mass - 75.337) > 1e-5:
        raise ValueError("ORIGINAL_SOURCE_MASS_MISMATCH")

    work = temp / name
    work.mkdir()
    geometry = path.parent / "Geometry"
    support = work / "support.json"
    stage("converter_support")
    rc_check = converter_command(["check", str(path), "--geometry", str(geometry),
                                  "--json", str(support)])
    if rc_check or not support.is_file():
        raise ValueError("PINNED_CONVERTER_PUBLIC_SUPPORT_CHECK_FAILED")
    report = json.loads(support.read_text(encoding="utf-8"))
    if report.get("outcome") != "converts":
        raise ValueError("PINNED_CONVERTER_UNSUPPORTED_ORIGINAL_ELEMENT")
    output = work / "converted"
    stage("stage1_conversion")
    rc_convert = converter_command(["convert", str(path), str(output),
                                    "--geometry", str(geometry), "--steps", "1"])
    if rc_convert:
        raise ValueError("PINNED_CONVERTER_STAGE1_FAILED")
    converted = output / (name + "_cvt1.xml")
    if not converted.is_file():
        raise ValueError("CONVERTED_NATIVE_XML_NOT_CREATED")

    # The stage-1 output is checked using real MuJoCo dynamics data and
    # real source OpenSim positions; third-party mesh assets are irrelevant
    # to body-origin kinematics, so compile without mesh geoms.
    stage("native_compile")
    native, _ = resolver.load_without_meshes(str(converted))
    if native.nbody <= 22:
        raise ValueError("NATIVE_BODY_STRUCTURE_INCOMPLETE")

    result = {
        "model": name, "source_commit": SOURCE_COMMIT,
        "converter_commit": CONVERTER_COMMIT, "source_blob_verified": True,
        "native_mujoco_compiled": True, "source_body_count": bodies.getSize(),
        "source_body_mass_kg": round(mass, 6),
        "support_counts": report.get("counts", {}),
    }
    stage("core_fk")
    result["core"] = verify_one_cohort(model, converted, CORE_KNEES, CORE_HIPS, "core")
    if name == "RajagopalLaiUhlrich2023":
        stage("2023_extension_fk")
        result["beyond_120deg"] = verify_one_cohort(
            model, converted, EXTRA_KNEES, EXTRA_HIPS, "2023_only")
    result["classification"] = "PASS" if (
        result["core"]["classification"] == "PASS" and
        result.get("beyond_120deg", {"classification": "PASS"})["classification"] == "PASS"
    ) else "FAIL"
    stage("completed")
    return result


def main() -> int:
    import mujoco
    import opensim
    if mujoco.__version__ != "3.15.0" or "4.6" not in str(opensim.GetVersion()):
        print("D4B_PUBLIC_STATUS VERSION_MISMATCH")
        return 2
    source = os.environ.get(SOURCE_ROOT_ENV)
    if not source:
        print("D4B_PUBLIC_STATUS SOURCE_ENV_MISSING")
        return 2
    with tempfile.TemporaryDirectory(prefix="public_d4b1_") as work:
        results = []
        for name, sha, size in MODELS:
            stage("source_identity")
            try:
                result = one_model(name, sha, size, Path(source), Path(work))
            except Exception as exc:
                # No raw model arrays, paths, XML or source-related exception text
                # in public logs; return only type and generic failure category.
                result = {"model": name, "classification": "ERROR",
                          "failure_stage": CURRENT_STAGE,
                          "error_type": type(exc).__name__}
            results.append(result)
            print("D4B_PUBLIC_MODEL_AGGREGATE", json.dumps(result, sort_keys=True))
    ok = (len(results) == len(MODELS)
          and all(x.get("classification") == "PASS" for x in results))
    print("D4B_PUBLIC_FINAL", "PASS" if ok else "FAIL",
          "original_sources_committed=false", "full_sts_validated=false",
          "patient_data_used=false")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
