"""Original-authored OpenSim 4.6 two-hinge converter capability pilot.

Source-independent synthetic fixture: no Rajagopal, no OpenSim model files,
no original muscle/spline/participant data, no private repository checkout.
"""
from __future__ import annotations

import json
import math
import pathlib
import subprocess
import sys
import xml.etree.ElementTree as ET


UPSTREAM_COMMIT="8c4d97e96f43c0d1676b8d58b0cff95af72a0f0f"
POSITIONS=((0.,0.),(.3,-.2),(.9,-.7))
MASS=(12.,4.)
NAMES=("proximal","distal")
DOFS=("q_hip","q_knee")


def build_model(filepath):
    import opensim as os
    p=pathlib.Path(filepath)
    p.parent.mkdir(parents=True,exist_ok=True)
    model=os.Model()
    model.setName("original_source_free_two_pin_converter_fixture")
    model.setGravity(os.Vec3(0.,-9.80665,0.))
    body0=os.Body("proximal",MASS[0],os.Vec3(0.,-.2,0.),os.Inertia(.25,.25,.10))
    body1=os.Body("distal",MASS[1],os.Vec3(0.,-.2,0.),os.Inertia(.08,.08,.03))
    model.addBody(body0)
    model.addBody(body1)
    hip=os.PinJoint("hip",model.getGround(),os.Vec3(0.,1.,0.),os.Vec3(0.),
                    body0,os.Vec3(0.,.2,0.),os.Vec3(0.))
    knee=os.PinJoint("knee",body0,os.Vec3(0.,-.2,0.),os.Vec3(0.),
                     body1,os.Vec3(0.,.2,0.),os.Vec3(0.))
    hip.updCoordinate().setName(DOFS[0])
    knee.updCoordinate().setName(DOFS[1])
    model.addJoint(hip)
    model.addJoint(knee)
    state=model.initSystem()
    if (model.getBodySet().getSize(),model.getJointSet().getSize())!=(2,2):
        raise AssertionError("OpenSim fixture wrong topology")
    for i,n in enumerate(NAMES):
        body=model.getBodySet().get(n)
        if abs(body.getMass()-MASS[i])>1e-12:
            raise AssertionError("OpenSim wrong body mass")
    # No real humans / original source model; mechanically connected native OpenSim fixture.
    model.printToXML(str(p))
    reread=os.Model(str(p))
    state2=reread.initSystem()
    if (reread.getBodySet().getSize(),reread.getJointSet().getSize())!=(2,2):
        raise AssertionError("source-free OpenSim XML roundtrip failed")
    return {"synthetic_opensim_bodies":2,"synthetic_opensim_joints":2,
            "source_mass_total_kg":sum(MASS),"tested_open_sim_file":p.name}


def run_command(args):
    proc=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                        encoding="utf-8",errors="replace",check=False,timeout=600)
    output=proc.stdout or ""
    print("COMMAND",args[0],args[1:3],"EXIT",proc.returncode)
    print("TOOL_OUTPUT_TAIL",output[-2600:])
    return proc.returncode,output


def discover_converted(out_dir):
    results=list(pathlib.Path(out_dir).glob("*_cvt1.xml"))
    if not results:
        results=[p for p in pathlib.Path(out_dir).glob("*.xml") if p.is_file()]
    if len(results)!=1:
        raise AssertionError(f"no unique converted native XML: {len(results)}")
    return results[0]


def check_compiled(out):
    import mujoco
    if mujoco.__version__!="3.15.0":
        raise AssertionError("not pinned converter's current native engine")
    xml=discover_converted(out)
    m=mujoco.MjModel.from_xml_path(str(xml.resolve()))
    if m.njnt!=2:
        raise AssertionError(f"unexpected joint count: {m.njnt}")
    if m.nv!=2:
        raise AssertionError(f"unexpected velocity DOF: {m.nv}")
    if any(m.jnt_type[i]==mujoco.mjtJoint.mjJNT_FREE for i in range(m.njnt)):
        raise AssertionError("freejoint shortcut is not a coupled 2-pin mechanism")
    found=[]
    for name,mass in zip(NAMES,MASS):
        bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,name)
        if bid<0:
            raise AssertionError(f"missing source body {name}")
        if abs(float(m.body_mass[bid])-mass)>1e-6:
            raise AssertionError(f"incorrect physical body mass {name}")
        found.append(bid)
    if (m.body_parentid[found[1]]!=found[0] or found[0]==0):
        raise AssertionError("not a properly connected parent-child tree")
    qids=[]
    for name in DOFS:
        jid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name)
        if jid<0:
            raise AssertionError(f"missing source joint coordinate {name}")
        qids.append(jid)
    # Verify actual MuJoCo FK on multiple source-independent angles and no invalid states.
    max_kinematic_radius=0.
    d=mujoco.MjData(m)
    for q0,q1 in POSITIONS:
        d.qpos[m.jnt_qposadr[qids[0]]]=q0
        d.qpos[m.jnt_qposadr[qids[1]]]=q1
        mujoco.mj_forward(m,d)
        for bid in found:
            if not all(math.isfinite(float(v)) for v in d.xpos[bid]):
                raise AssertionError("native pose nonfinite")
            max_kinematic_radius=max(max_kinematic_radius,
                                     math.sqrt(float(sum(x*x for x in d.xpos[bid]))))
    return {"native_mujoco_version":mujoco.__version__,
            "native_body_count":m.nbody,"native_joint_count":m.njnt,
            "native_velocity_dof":m.nv,"source_named_bodies_found":len(found),
            "source_named_joint_coordinates_found":len(qids),
            "source_mass_kg":sum(float(m.body_mass[i]) for i in found),
            "source_tree_connected":True,
            "three_native_pose_samples_finite":True,
            "max_native_body_distance_from_world_origin_m":max_kinematic_radius,
            "source_vs_native_pose_equivalence":"NOT_TESTED_IN_THIS_G0_G4_PILOT"}


def main():
    import opensim, mujoco
    import tempfile
    print("VERSIONS",getattr(opensim,"__version__",None),
          getattr(opensim,"GetVersion",lambda:"unknown")(),mujoco.__version__)
    if mujoco.__version__!="3.15.0":
        raise AssertionError("converter MuJoCo runtime mismatch")
    with tempfile.TemporaryDirectory(prefix="parkour_synthetic_converter_") as temp:
        t=pathlib.Path(temp)
        model=t/"toy_two_pin.osim"
        source=build_model(model)
        report=t/"check.json"
        check_code,stdout=run_command(["o2m","check",str(model),"--json",str(report)])
        if check_code!=0:
            raise AssertionError("o2m check did not predict convertibility")
        if not report.is_file() or report.stat().st_size<100:
            raise AssertionError("empty or missing independent capability report")
        check_json=json.loads(report.read_text())
        print("CHECK_REPORT_TYPE",type(check_json).__name__,
              "TOP_LEVEL_KEYS",list(check_json)[:20] if isinstance(check_json,dict) else "list")
        out=t/"converted"
        convert_code,stdout=run_command(["o2m","convert",str(model),str(out),
                                          "--steps","1"])
        if convert_code!=0:
            raise AssertionError("converter failed real XML stage")
        converted=check_compiled(out)
        record={"experiment":"SIM-026C4B2D4A",
                "scope":"completely independent source-free OpenSim 2-pin conversion",
                "upstream_converter_pinned_sha":UPSTREAM_COMMIT,
                "source_model":source,
                "converter_check_exit_code":check_code,
                "converter_check_report_bytes":report.stat().st_size,
                "converter_stage1_exit_code":convert_code,
                "native_compilation_and_topology":converted,
                "native_source_matched_pose_G5":"NOT_EXECUTED_YET",
                "real_Rajagopal_source_evaluation":"NO_GO",
                "original_source_redistribution_rights":"UNRESOLVED",
                "3contact_3D_STS_or_human_orthosis":"NO_GO"}
        print("AGGREGATE_JSON",json.dumps(record,sort_keys=True))
        return record


if __name__=="__main__":
    main()
