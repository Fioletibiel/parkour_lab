"""D4B2-P1a: stage-1 *public original* OpenSim vs native MuJoCo muscle-path screen.

Runs only on already-published upstream example models checked out transiently.
Source model, geometry, converted MJCF and per-pose muscle quantities remain
in the CI job memory/private temp workspace; only aggregate statistics print.
No private Parkour files, participant data or clinical claims.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import tempfile

from lab import d4b1_open_model_native as p0

COORDS = ("knee_angle_r", "knee_angle_l", "hip_flexion_r", "hip_flexion_l")
FD_H = 1e-6
FD_CHECK_H = 1e-5
FD_SELF_CONSISTENCY_LIMIT_M = 1e-6
FD_SELF_CONSISTENCY_FRACTION = .99
MIN_SOURCE_ARM_TO_SCORE_M = .001
N_MUSCLES = 80
N_RUNTIME_JOINTS = 22
N_BODIES = 22
N_COORDS = 39
MASS_KG = 75.337
REPORT_PREFIX = "D4B2_PUBLIC_MUSCLE"
STAGE = "preflight"


def stage(value):
    global STAGE
    allowed = {
        "source_identity", "source_load", "source_structure", "stage1_convert",
        "native_compile", "muscle_mapping", "core_measure", "extension_measure",
        "aggregate",
    }
    if value not in allowed:
        raise ValueError("UNRECOGNIZED_STAGE")
    STAGE = value


def finite_nonnegative(x, label):
    x = float(x)
    if not math.isfinite(x) or x < 0:
        raise ValueError("BAD_MEASURED_" + label)
    return x


def safe_fraction_good(delta, threshold=FD_SELF_CONSISTENCY_LIMIT_M):
    """Finite differences must be finite; do not make NaN imply a PASS."""
    import numpy as np
    x = np.asarray(delta, dtype=float)
    if x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("INVALID_FINITE_DIFFERENCE_ARRAY")
    return float(np.mean(np.abs(x) <= threshold))


def summarize_errors(source_lengths, native_lengths, source_ma, native_ma,
                     source_ma_check, native_ma_check, cohort):
    """Pure numeric aggregator; does not change prereg thresholds."""
    import numpy as np
    os_len = np.asarray(source_lengths, dtype=float)
    mj_len = np.asarray(native_lengths, dtype=float)
    osm = np.asarray(source_ma, dtype=float)
    mjm = np.asarray(native_ma, dtype=float)
    os10 = np.asarray(source_ma_check, dtype=float)
    mj10 = np.asarray(native_ma_check, dtype=float)
    if os_len.shape != mj_len.shape or len(os_len.shape) != 2 or os_len.shape[1] != N_MUSCLES:
        raise ValueError("INVALID_MUSCLE_LENGTH_COVERAGE")
    if len(os_len) not in (2, 4):
        raise ValueError("INVALID_POSE_COUNT")
    expected = os_len.shape + (len(COORDS),)
    if any(v.shape != expected for v in (osm, mjm, os10, mj10)):
        raise ValueError("INVALID_MOMENT_ARM_COVERAGE")
    if any(not np.all(np.isfinite(v)) for v in (os_len, mj_len, osm, mjm, os10, mj10)):
        raise ValueError("NONFINITE_MUSCLE_OBSERVATIONS")
    if np.min(os_len) < 0 or np.min(mj_len) < 0:
        raise ValueError("NEGATIVE_MUSCULOTENDON_LENGTH")

    errors_mm = 1000. * (mj_len - os_len)
    pair_peak = np.max(np.abs(osm), axis=0)
    scored = pair_peak >= MIN_SOURCE_ARM_TO_SCORE_M
    if not np.any(scored):
        raise ValueError("NO_SPANNING_MUSCLE_COORDINATE_PAIRS")
    pair_rms_mm = 1000. * np.sqrt(np.mean((mjm - osm) ** 2, axis=0))
    per_muscle_len_rms_mm = np.sqrt(np.mean(errors_mm ** 2, axis=0))
    valid_pair_rms = pair_rms_mm[scored]
    scored_source_peaks = 1000. * pair_peak[scored]
    rms_pct = valid_pair_rms / scored_source_peaks * 100.
    source_fd_frac = safe_fraction_good(osm - os10)
    native_fd_frac = safe_fraction_good(mjm - mj10)
    coverage_ok = (source_fd_frac >= FD_SELF_CONSISTENCY_FRACTION and
                   native_fd_frac >= FD_SELF_CONSISTENCY_FRACTION)
    return {
        "cohort": cohort, "poses": int(len(os_len)),
        "original_muscles": int(os_len.shape[1]),
        "source_native_lengths_compared": int(os_len.size),
        "coordinate_derivative_samples": int(osm.size),
        "muscle_coordinate_pairs_total": int(pair_peak.size),
        "scored_pairs_above_1mm_source_peak": int(np.count_nonzero(scored)),
        "source_zero_or_small_pairs": int(np.count_nonzero(~scored)),
        "path_length_rms_mm_overall": round(float(np.sqrt(np.mean(errors_mm ** 2))), 8),
        "path_length_rms_mm_median_muscle": round(float(np.median(per_muscle_len_rms_mm)), 8),
        "path_length_rms_mm_p95_muscle": round(float(np.percentile(per_muscle_len_rms_mm, 95)), 8),
        "path_length_rms_mm_worst_muscle": round(float(np.max(per_muscle_len_rms_mm)), 8),
        "path_length_abs_max_mm": round(float(np.max(np.abs(errors_mm))), 8),
        "moment_arm_rms_mm_mean_scored_pair": round(float(np.mean(valid_pair_rms)), 8),
        "moment_arm_rms_mm_median_scored_pair": round(float(np.median(valid_pair_rms)), 8),
        "moment_arm_rms_mm_worst_scored_pair": round(float(np.max(valid_pair_rms)), 8),
        "moment_arm_rms_pct_source_peak_mean": round(float(np.mean(rms_pct)), 6),
        "scored_pairs_within_2mm_RMS": int(np.count_nonzero(valid_pair_rms <= 2.)),
        "scored_pairs_within_5pct_source_peak_RMS": int(np.count_nonzero(rms_pct <= 5.)),
        "source_fd_h_vs_10h_consistency_fraction": round(source_fd_frac, 8),
        "native_fd_h_vs_10h_consistency_fraction": round(native_fd_frac, 8),
        "classification": "SCREEN_VALID_MEASURED" if coverage_ok else "SCREEN_INVALID_FD",
        "absolute_muscle_accuracy_gate": "NOT_PREDECLARED",
    }


def setup_model(source_path, converted_path):
    import opensim
    import mujoco
    from osim2mujoco.checks import mujoco_compare as mc

    original = opensim.Model(str(source_path))
    state = original.initSystem()
    coord_set = original.getCoordinateSet()
    coord_names = [coord_set.get(i).getName() for i in range(coord_set.getSize())]
    coords = {n:coord_set.get(n) for n in coord_names}
    if not set(COORDS).issubset(coords):
        raise ValueError("REQUIRED_SOURCE_COORDINATES_ABSENT")
    source_default = {n:float(coords[n].getDefaultValue()) for n in coord_names}
    muscles = original.getMuscles()
    source_names = [muscles.get(i).getName() for i in range(muscles.getSize())]
    if len(source_names) != N_MUSCLES or len(set(source_names)) != N_MUSCLES:
        raise ValueError("SOURCE_MUSCLE_COUNT_OR_DUPLICATE")
    if (original.getBodySet().getSize(), original.getJointSet().getSize(),
            original.getCoordinateSet().getSize()) != (N_BODIES,N_RUNTIME_JOINTS,N_COORDS):
        raise ValueError("SOURCE_RUNTIME_TOPOLOGY")
    if abs(sum(original.getBodySet().get(i).getMass()
               for i in range(N_BODIES)) - MASS_KG)>1e-5:
        raise ValueError("SOURCE_BODY_MASS")
    native,_ = mc.load_without_meshes(str(converted_path))
    data = mujoco.MjData(native)
    direct, driven, problems = mc.joint_plan(native, coord_names)
    if problems:
        raise ValueError("INVALID_NATIVE_JOINT_DEPENDENCY_PLAN")
    source_couplers = []
    cs=original.getConstraintSet()
    for i in range(cs.getSize()):
        cc=opensim.CoordinateCouplerConstraint.safeDownCast(cs.get(i))
        if cc is None:
            raise ValueError("UNSUPPORTED_SOURCE_CONSTRAINT_TYPE")
        if cc.get_isEnforced():
            indep=cc.getIndependentCoordinateNames()
            source_couplers.append((cc.getDependentCoordinateName(),
                                    [indep.get(j) for j in range(indep.getSize())],cc))

    tids=[]
    for name in source_names:
        tid=mujoco.mj_name2id(native,mujoco.mjtObj.mjOBJ_TENDON,name+"_tendon")
        aid=mujoco.mj_name2id(native,mujoco.mjtObj.mjOBJ_ACTUATOR,name)
        if tid < 0 or aid < 0:
            raise ValueError("MISSING_NATIVE_NAMED_TENDON_OR_ACTUATOR")
        if (int(native.actuator_trntype[aid]) != int(mujoco.mjtTrn.mjTRN_TENDON)
                or int(native.actuator_trnid[aid,0]) != tid):
            raise ValueError("NATIVE_ACTUATOR_WRONG_TENDON")
        if int(native.actuator_dyntype[aid]) != int(mujoco.mjtDyn.mjDYN_MUSCLE):
            raise ValueError("NATIVE_ACTUATOR_NOT_MUSCLE")
        tids.append(tid)
    import numpy as np
    active=native.eq_active0 if hasattr(native,"eq_active0") else native.eq_active
    active_eq=int(np.count_nonzero(active))
    if active_eq <= 0:
        raise ValueError("NO_ACTIVE_NATIVE_JOINT_EQUALITIES")
    return (original,state,coords,coord_names,source_default,muscles,
            native,data,direct,driven,source_couplers,np.array(tids),active_eq)


def measure_cohort(context, knees, hips, cohort):
    import opensim
    import numpy as np
    from osim2mujoco.checks import mujoco_compare as mc

    (original,state,coords,coord_names,defaults,muscles,
     native,data,direct,driven,couplers,tids,n_eq)=context
    if len(knees) not in (2,4) or len(knees)!=len(hips):
        raise ValueError("INCONSISTENT_CORE_OR_EXTENSION_POSES")
    src_l=[]; mj_l=[]; src_ma=[]; mj_ma=[]; src_ma10=[]; mj_ma10=[]
    max_src_qerr=max_native_res=0.0
    source_passes=native_passes=0

    def observe(values):
        nonlocal max_src_qerr,max_native_res,source_passes,native_passes
        for dependent,ind,cc in couplers:
            x=opensim.Vector(len(ind),0.)
            for i,n in enumerate(ind):
                x.set(i,values[n])
            values[dependent]=float(cc.getFunction().calcValue(x))
        for name in coord_names:
            c=coords[name]
            if c.getLocked(state):
                if abs(float(c.getValue(state))-values[name])>1e-10:
                    raise ValueError("LOCKED_COORDINATE_NOT_DEFAULT")
                continue
            c.setValue(state,values[name],False)
        original.realizePosition(state)
        qerr=state.getQErr()
        for k in range(qerr.size()):
            max_src_qerr=max(max_src_qerr,finite_nonnegative(
                abs(qerr.get(k)),"SOURCE_QERR"))
        if max_src_qerr>p0.QERR_LIMIT:
            raise ValueError("SOURCE_CONSTRAINT_QERR_EXCEEDED")
        source_passes+=1
        source=np.array([muscles.get(i).getLength(state)
                         for i in range(N_MUSCLES)],dtype=float)
        mc.set_pose(native,data,direct,driven,values)
        residual,rows=mc.equality_residual(native,data)
        max_native_res=max(max_native_res,finite_nonnegative(
            residual,"NATIVE_EQUALITY_RESIDUAL"))
        if rows != n_eq or max_native_res>p0.EQ_LIMIT:
            raise ValueError("MISSING_OR_INVALID_NATIVE_EQUALITIES")
        native_passes+=1
        native_lengths=np.array(data.ten_length[tids],dtype=float)
        if (not np.all(np.isfinite(source)) or not np.all(np.isfinite(native_lengths))
                or np.min(source)<0 or np.min(native_lengths)<0):
            raise ValueError("INVALID_SOURCE_OR_NATIVE_MUSCLE_LENGTHS")
        return source,native_lengths

    for knee,hip in zip(knees,hips):
        base=dict(defaults)
        base.update(zip(("knee_angle_r","knee_angle_l"),knee))
        base.update(zip(("hip_flexion_r","hip_flexion_l"),hip))
        a,b=observe(dict(base))
        src_l.append(a);mj_l.append(b)
        deriv_s=[];deriv_n=[];deriv_s10=[];deriv_n10=[]
        for coord in COORDS:
            pair={}
            for h in (FD_H,FD_CHECK_H):
                plus=dict(base);plus[coord]+=h
                minus=dict(base);minus[coord]-=h
                sp,mp=observe(plus);sm,mm=observe(minus)
                pair[h]=(-(sp-sm)/(2*h),-(mp-mm)/(2*h))
            deriv_s.append(pair[FD_H][0]);deriv_n.append(pair[FD_H][1])
            deriv_s10.append(pair[FD_CHECK_H][0]);deriv_n10.append(pair[FD_CHECK_H][1])
        src_ma.append(np.stack(deriv_s,axis=1))
        mj_ma.append(np.stack(deriv_n,axis=1))
        src_ma10.append(np.stack(deriv_s10,axis=1))
        mj_ma10.append(np.stack(deriv_n10,axis=1))

    stage("aggregate")
    summary=summarize_errors(np.stack(src_l),np.stack(mj_l),
                             np.stack(src_ma),np.stack(mj_ma),
                             np.stack(src_ma10),np.stack(mj_ma10),cohort)
    summary.update({
        "max_source_qerr":max_src_qerr,
        "max_native_equality_residual":max_native_res,
        "native_active_equalities":n_eq,
        "source_realize_calls":source_passes,
        "native_forward_calls":native_passes,
    })
    return summary


def run_one(name,blob,size,source_root,temporary):
    import opensim
    stage("source_identity")
    path=source_root/"Models"/"Rajagopal"/(name+".osim")
    if path.stat().st_size != size or p0.model_git_sha1(path)!=blob:
        raise ValueError("PINNED_OFFICIAL_PUBLIC_SOURCE_MISMATCH")
    stage("source_load")
    source=opensim.Model(str(path))
    source.initSystem()
    stage("source_structure")
    if (source.getMuscles().getSize()!=N_MUSCLES or
            source.getBodySet().getSize()!=N_BODIES or
            source.getJointSet().getSize()!=N_RUNTIME_JOINTS or
            source.getCoordinateSet().getSize()!=N_COORDS):
        raise ValueError("SOURCE_MODEL_STRUCTURE_INVALID")
    stage("stage1_convert")
    out=temporary/name
    out.mkdir()
    geometry=path.parent/"Geometry"
    if p0.converter_command(["convert",str(path),str(out),
                             "--geometry",str(geometry),"--steps","1"],timeout=1200)!=0:
        raise ValueError("ORIGINAL_PUBLIC_MODEL_STAGE1_CONVERSION_FAILED")
    xml=out/(name+"_cvt1.xml")
    if not xml.is_file():
        raise ValueError("NO_STAGE1_CONVERTED_XML")
    stage("native_compile")
    context=setup_model(path,xml)
    stage("muscle_mapping")
    result={"model":name,"source_blob_verified":True,
            "original_muscle_count":N_MUSCLES,
            "converter_stage":"1_XML_ONLY_NOT_FITTED"}
    stage("core_measure")
    result["core"]=measure_cohort(context,p0.CORE_KNEES,p0.CORE_HIPS,"core")
    if name=="RajagopalLaiUhlrich2023":
        stage("extension_measure")
        result["beyond_120deg"]=measure_cohort(
            context,p0.EXTRA_KNEES,p0.EXTRA_HIPS,"2023_only")
    valid=all(x["classification"]=="SCREEN_VALID_MEASURED"
              for x in (result[k] for k in ("core","beyond_120deg") if k in result))
    result["classification"]="SCREEN_VALID_MEASURED" if valid else "SCREEN_INVALID_FD"
    return result


def main():
    import opensim
    import mujoco
    if mujoco.__version__!="3.15.0" or "4.6" not in str(opensim.GetVersion()):
        print(REPORT_PREFIX+"_ENVIRONMENT VERSION_REJECTED")
        return 2
    source=os.environ.get(p0.SOURCE_ROOT_ENV)
    if not source:
        print(REPORT_PREFIX+"_ENVIRONMENT PUBLIC_SOURCE_NOT_MOUNTED")
        return 2
    results=[]
    with tempfile.TemporaryDirectory(prefix="public_d4b2_p1a_") as dirname:
        for name,blob,size in p0.MODELS:
            stage("source_identity")
            try:
                output=run_one(name,blob,size,Path(source),Path(dirname))
            except Exception as exc:
                output={"model":name,"classification":"ERROR",
                        "failure_stage":STAGE,"error_type":type(exc).__name__}
            results.append(output)
            print(REPORT_PREFIX+"_MODEL_AGGREGATE",json.dumps(output,sort_keys=True))
    ok=len(results)==2 and all(r.get("classification")=="SCREEN_VALID_MEASURED"
                               for r in results)
    print(REPORT_PREFIX+"_FINAL",("SCREEN_VALID_MEASURED" if ok else "SCREEN_INVALID"),
          "private_sources_used=false", "force_accuracy_validated=false",
          "full_sts_validated=false")
    return 0 if ok else 1


if __name__=="__main__":
    sys.exit(main())
