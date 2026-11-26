"""D4B2-P1b: compare stage-1 and stage-2 muscle paths on one public source.

Uses pinned original publicly published RajagopalLaiUhlrich2023, no patient
data, no private Parkour checkout. All model/converted data stay ephemeral.
This is a before/after static geometry screen, NOT force/ST S validation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile

from lab import d4b1_open_model_native as p0
from lab import d4b2_muscle_path_screen as p1

MODEL_NAME="RajagopalLaiUhlrich2023"
BASELINE_STAGE1_CORE_WORST_LENGTH_MM=203.95778123
BASELINE_STAGE1_EXTRA_WORST_LENGTH_MM=9.44103028
BASELINE_REPEAT_TOL_MM=0.01
SCREEN_STATUS="SCREEN_VALID_MEASURED"
STAGE="preflight"


def stage(name):
    global STAGE
    names={"source_identity","source_structure","stage2_conversion",
           "xml_check","stage1_core","stage1_extra",
           "stage2_core","stage2_extra","comparison"}
    if name not in names:
        raise ValueError("UNKNOWN_STAGE")
    STAGE=name


def compare(before,after):
    """Only comparative numerical deltas; NEVER assert anatomical force accuracy."""
    keys=("path_length_rms_mm_overall","path_length_rms_mm_median_muscle",
          "path_length_rms_mm_p95_muscle","path_length_rms_mm_worst_muscle",
          "path_length_abs_max_mm","moment_arm_rms_mm_mean_scored_pair",
          "moment_arm_rms_mm_median_scored_pair",
          "moment_arm_rms_mm_worst_scored_pair",
          "moment_arm_rms_pct_source_peak_mean")
    if before["classification"]!=SCREEN_STATUS or after["classification"]!=SCREEN_STATUS:
        raise ValueError("INVALID_SOURCE_NATIVE_COMPARISON_COHORT")
    if (before["source_native_lengths_compared"]!=after["source_native_lengths_compared"]
            or before["scored_pairs_above_1mm_source_peak"]!=
                    after["scored_pairs_above_1mm_source_peak"]):
        raise ValueError("DIFFERENT_COMPARISON_COVERAGE")
    delta={k:round(after[k]-before[k],8) for k in keys}
    return {
        "cohort":before["cohort"],
        "same_full_80_muscle_coverage":True,
        "error_change_mm_or_pct_stage2_minus_stage1":delta,
        "max_path_length_error_improved":delta["path_length_abs_max_mm"]<0,
        "max_scored_moment_arm_rms_improved":delta["moment_arm_rms_mm_worst_scored_pair"]<0,
        "muscle_accuracy_pass":"NOT_ESTABLISHED",
    }


def run():
    from pathlib import Path
    import opensim
    import mujoco

    if mujoco.__version__!="3.15.0" or "4.6" not in str(opensim.GetVersion()):
        raise ValueError("WRONG_ENGINE_VERSIONS")
    root_str=os.environ.get(p0.SOURCE_ROOT_ENV)
    if not root_str:
        raise ValueError("MISSING_PUBLIC_OPEN_MODEL_SOURCE_ROOT")
    root=Path(root_str)
    entry=next((x for x in p0.MODELS if x[0]==MODEL_NAME),None)
    if entry is None:
        raise ValueError("FROZEN_MODEL_REFERENCE_MISSING")
    name,blob,size=entry
    source=root/"Models"/"Rajagopal"/(name+".osim")
    stage("source_identity")
    if source.stat().st_size != size or p0.model_git_sha1(source)!=blob:
        raise ValueError("ORIGINAL_PUBLIC_MODEL_GIT_BLOB_MISMATCH")
    stage("source_structure")
    source_model=opensim.Model(str(source))
    source_model.initSystem()
    if (source_model.getMuscles().getSize(),source_model.getBodySet().getSize(),
            source_model.getJointSet().getSize(),
            source_model.getCoordinateSet().getSize())!=(80,22,22,39):
        raise ValueError("ORIGINAL_PUBLIC_MODEL_TOPOLOGY_MISMATCH")
    with tempfile.TemporaryDirectory(prefix="public_d4b2_p1b_") as dirname:
        temp=Path(dirname)
        output=temp/"stage12"
        stage("stage2_conversion")
        code=p0.converter_command(["convert",str(source),str(output),
                                    "--geometry",str(source.parent/"Geometry"),
                                    "--steps","1","2"],timeout=2400)
        if code!=0:
            raise ValueError("PINNED_STAGE2_CONVERSION_UNSUCCESSFUL")
        stage("xml_check")
        cvt1=output/(name+"_cvt1.xml")
        cvt2=output/(name+"_cvt2.xml")
        if not cvt1.is_file() or not cvt2.is_file():
            raise ValueError("STAGE1_OR_STAGE2_CONVERTED_XML_MISSING")
        context1=p1.setup_model(source,cvt1)
        context2=p1.setup_model(source,cvt2)
        stage("stage1_core")
        base_core=p1.measure_cohort(context1,p0.CORE_KNEES,p0.CORE_HIPS,"core")
        stage("stage1_extra")
        base_ext=p1.measure_cohort(context1,p0.EXTRA_KNEES,p0.EXTRA_HIPS,"2023_only")
        if (abs(base_core["path_length_abs_max_mm"] -
                BASELINE_STAGE1_CORE_WORST_LENGTH_MM)>BASELINE_REPEAT_TOL_MM or
            abs(base_ext["path_length_abs_max_mm"] -
                BASELINE_STAGE1_EXTRA_WORST_LENGTH_MM)>BASELINE_REPEAT_TOL_MM):
            raise ValueError("FROZEN_STAGE1_BASELINE_DID_NOT_REPRODUCE")

        stage("stage2_core")
        fit_core=p1.measure_cohort(context2,p0.CORE_KNEES,p0.CORE_HIPS,"core")
        stage("stage2_extra")
        fit_ext=p1.measure_cohort(context2,p0.EXTRA_KNEES,p0.EXTRA_HIPS,"2023_only")
        stage("comparison")
        compared_core=compare(base_core,fit_core)
        compared_ext=compare(base_ext,fit_ext)
        return {
            "experiment":"D4B2-P1b", "model":MODEL_NAME,
            "original_public_model_commit":p0.SOURCE_COMMIT,
            "public_converter_commit":p0.CONVERTER_COMMIT,
            "stage1_reproduced_within_0p01_mm":True,
            "all_80_muscles_in_both_stages":True,
            "core":{"stage1":base_core,"stage2":fit_core,"comparison":compared_core},
            "beyond_120deg":{"stage1":base_ext,"stage2":fit_ext,"comparison":compared_ext},
            "classification":SCREEN_STATUS,
            "muscle_force_validated":False,"dynamic_sts_validated":False,
            "private_sources_used":False,
        }


def main():
    try:
        result=run()
    except Exception as exc:
        # Public logs never expose original XML, paths, anatomical vectors
        # or detailed Python exception strings.
        print("D4B2_STAGE2_FINAL",json.dumps({
            "classification":"ERROR",
            "failure_stage":STAGE,"error_type":type(exc).__name__}))
        return 1
    print("D4B2_STAGE2_AGGREGATE",json.dumps(result,sort_keys=True))
    print("D4B2_STAGE2_FINAL",SCREEN_STATUS,
          "muscle_accuracy_pass=false","forces_validated=false")
    return 0


if __name__=="__main__":
    sys.exit(main())
