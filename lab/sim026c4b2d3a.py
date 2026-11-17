"""SIM-026C4B2D3A: wholly synthetic switched piecewise-polynomial native constraint.

Never derived from original OpenSim anatomy, source splines or private research data.
Only a 2-body toy hinge/slide, not a real knee or support-device simulation.
"""
from __future__ import annotations

import json
import math

import numpy as np

VERSION="3.3.6"
KNOT=.7
LOW=(0.,.03,.02,0.,0.)
HIGH=(-.00686,.0594,-.022,.02,0.)
SAMPLES=(0.,.15,.4,.69,.699999,.7,.700001,.71,.9,1.2,1.6)
STEPS=500
DT=.001
TORQUE=.1
XML="""<mujoco model="public_synthetic_piecewise_connected">
  <compiler angle="radian"/>
  <option timestep="0.001" gravity="0 0 0" integrator="implicitfast"/>
  <worldbody>
    <body name="parent">
      <joint name="q" type="hinge" axis="0 0 1" damping="0.2"/>
      <inertial pos="0.1 0 0" mass="1.1" diaginertia="0.04 0.04 0.02"/>
      <body name="child" pos="1 0 0">
        <joint name="s" type="slide" axis="1 0 0" damping="0.2"/>
        <inertial pos="0.2 0 0" mass="0.6" diaginertia="0.01 0.02 0.02"/>
      </body>
    </body>
  </worldbody>
  <equality>
    <joint name="low" joint1="s" joint2="q" polycoef="0 .03 .02 0 0"
           active="true" solref="0.005 1" solimp="0.99 0.99 0.001"/>
    <joint name="high" joint1="s" joint2="q" polycoef="-.00686 .0594 -.022 .02 0"
           active="false" solref="0.005 1" solimp="0.99 0.99 0.001"/>
  </equality>
</mujoco>"""


def eval_poly(coef,q,derivative=0):
    if derivative not in (0,1,2,3):
        raise ValueError("unsupported derivative")
    if not math.isfinite(q):
        raise ValueError("invalid coordinate")
    return sum(math.prod(range(k-derivative+1,k+1))*coef[k]*q**(k-derivative)
               for k in range(derivative,5))


def target(q,derivative=0):
    q=float(q)
    if not math.isfinite(q) or q<0 or q>1.6:
        raise ValueError("frozen synthetic q domain violated")
    return eval_poly(LOW if q<=KNOT else HIGH,q,derivative)


def check_continuity():
    errs=[abs(eval_poly(LOW,KNOT,k)-eval_poly(HIGH,KNOT,k)) for k in (0,1,2)]
    jump=eval_poly(HIGH,KNOT,3)-eval_poly(LOW,KNOT,3)
    if max(errs)>1e-12 or abs(jump-.12)>1e-12:
        raise AssertionError("synthetic knot C2 / C3 jump changed")
    return {"max_C2_disagreement":max(errs),"third_derivative_jump":jump}


def global_quartic_failure():
    train=np.array((0.,.25,.5,.75,1.))
    test=np.array((.125,.375,.625,.875,1.125,1.375))
    vals=np.array([target(float(q)) for q in train])
    a=np.linalg.solve(np.vander(train,5,increasing=True),vals)
    errors=[abs(float(np.polynomial.polynomial.polyval(q,a))-target(float(q))) for q in test]
    worst=max(errors)
    if worst<=1e-7:
        raise AssertionError("synthetic piecewise function unexpectedly globally quartic")
    return {"train":len(train),"heldout":len(test),"max_global_quartic_holdout_error_m":worst}


def verify_xml_fixture(xml=XML):
    import xml.etree.ElementTree as ET
    root=ET.fromstring(xml)
    if root.tag!="mujoco" or root.get("model")!="public_synthetic_piecewise_connected":
        raise ValueError("wrong synthetic model identity")
    bodies=root.findall("./worldbody/body")
    if len(bodies)!=1 or bodies[0].get("name")!="parent":
        raise ValueError("wrong parent")
    children=bodies[0].findall("body")
    if len(children)!=1 or children[0].get("name")!="child" or children[0].get("pos")!="1 0 0":
        raise ValueError("wrong child or geometry")
    for node,name,kind,axis in [(bodies[0],"q","hinge","0 0 1"),(children[0],"s","slide","1 0 0")]:
        joints=node.findall("joint")
        if (len(joints)!=1 or joints[0].get("name")!=name or
           joints[0].get("type")!=kind or joints[0].get("axis")!=axis):
            raise ValueError("wrong joint")
        if len(node.findall("inertial"))!=1 or node.findall("geom") or node.findall("freejoint"):
            raise ValueError("missing inertia/unsupported geometry")
    eq=root.findall("./equality/joint")
    if len(eq)!=2 or [e.get("name") for e in eq]!=["low","high"]:
        raise ValueError("not two switched equality constraints")
    for node,poly,active in zip(eq,(LOW,HIGH),("true","false")):
        if ((node.get("joint1"),node.get("joint2"),node.get("active"))!=("s","q",active)
            or tuple(float(x) for x in node.get("polycoef","").split())!=poly):
            raise ValueError("wrong frozen equality coefficients")
    return True


def compiled():
    import mujoco as mu
    if mu.__version__!=VERSION:
        raise ValueError("wrong engine version")
    verify_xml_fixture()
    m=mu.MjModel.from_xml_string(XML)
    if (m.nbody,m.njnt,m.nq,m.nv,m.neq,m.ngeom)!=(3,2,2,2,2,0):
        raise ValueError("wrong native topology")
    if (tuple(m.eq_data[0,:5])!=LOW or tuple(m.eq_data[1,:5])!=HIGH or
        not np.allclose(m.body_mass[1:],(1.1,.6),atol=1e-12)):
        raise ValueError("native source-free masses/constraints altered")
    if abs(m.opt.timestep-DT)>1e-15 or np.linalg.norm(m.opt.gravity)>1e-15:
        raise ValueError("invalid dynamic environment")
    if (mu.mj_name2id(m,mu.mjtObj.mjOBJ_JOINT,"q"),
        mu.mj_name2id(m,mu.mjtObj.mjOBJ_JOINT,"s"))!=(0,1):
        raise ValueError("wrong joint coordinate order")
    return mu,m


def select_active(d,q):
    if not math.isfinite(float(q)):
        raise ValueError("nonfinite q cannot switch")
    low=1 if q<=KNOT else 0
    high=1-low
    d.eq_active[:]=[low,high]
    if sum(int(x) for x in d.eq_active)!=1:
        raise AssertionError("must have exactly one native equality")
    return int(high)


def static_native(mu,m):
    d=mu.MjData(m)
    bid=mu.mj_name2id(m,mu.mjtObj.mjOBJ_BODY,"child")
    max_p=max_a=max_c=0.
    observed=[]
    for q in SAMPLES:
        d.qpos[:]=[q,target(q)]
        d.qvel[:]=0
        high=select_active(d,q)
        mu.mj_forward(m,d)
        position=np.array([(1+target(q))*math.cos(q),(1+target(q))*math.sin(q),0.])
        rot=np.array([[math.cos(q),-math.sin(q),0],
                      [math.sin(q),math.cos(q),0],[0,0,1.]])
        actual=d.xmat[bid].reshape(3,3)
        rel=rot.T@actual
        vec=np.array((rel[2,1]-rel[1,2],rel[0,2]-rel[2,0],rel[1,0]-rel[0,1]))
        angle=math.atan2(float(np.linalg.norm(vec))*.5,float(np.trace(rel)-1)*.5)
        ep=float(np.linalg.norm(d.xpos[bid]-position))
        ec=float(np.max(np.abs(d.efc_pos))) if d.nefc else float("inf")
        if ep>1e-10 or angle>1e-10 or ec>1e-8 or d.nefc!=1:
            raise AssertionError("native switched piecewise joint FK/constraint failure")
        max_p=max(max_p,ep);max_a=max(max_a,angle);max_c=max(max_c,ec)
        observed.append(high)
    if sum(observed)!=5:
        raise AssertionError("test does not exercise both branches correctly")
    return {"poses":len(SAMPLES),"high_branch_poses":sum(observed),
            "max_pose_error_m":max_p,"max_rotation_error_rad":max_a,
            "max_constraint_error_m":max_c}


def native_dynamics(mu,m,driven=False):
    d=mu.MjData(m)
    if driven:
        q0=.68
        d.qpos[:]=[q0,target(q0)]
        d.qvel[:]=[.8,target(q0,1)*.8]
    mu.mj_forward(m,d)
    initial=int(d.eq_active[1])
    switches=0
    peak_residual=peak_speed=peak_force=peak_delta_q=peak_delta_s=0.
    prevq,prevs=(float(x) for x in d.qpos)
    for i in range(STEPS):
        before=int(d.eq_active[1])
        now=select_active(d,float(d.qpos[0]))
        switches+=int(now!=before)
        d.qfrc_applied[:]=[TORQUE if driven else 0.,0.]
        mu.mj_step(m,d)
        if any(w.number for w in d.warning):
            raise AssertionError("native MuJoCo warning")
        if not all(np.all(np.isfinite(v)) for v in (d.qpos,d.qvel,d.xpos,d.efc_force)):
            raise AssertionError("nonfinite switched dynamics")
        q,s=(float(x) for x in d.qpos)
        if q<0 or q>1.6:
            raise AssertionError("source-free domain exceeded")
        err=abs(s-target(q))
        if i>=25 and err>.005:
            raise AssertionError("piecewise equality native dynamic divergence")
        peak_residual=max(peak_residual,err)
        peak_speed=max(peak_speed,float(np.max(np.abs(d.qvel))))
        peak_force=max(peak_force,float(np.max(np.abs(d.efc_force))) if d.nefc else 0.)
        peak_delta_q=max(peak_delta_q,abs(q-prevq))
        peak_delta_s=max(peak_delta_s,abs(s-prevs))
        prevq,prevs=q,s
        if peak_speed>=25:
            raise AssertionError("unexpected native speed")
    if driven:
        if switches<1 or not .7<d.qpos[0]<1.6:
            raise AssertionError("driven dynamic run did not switch through knot")
    elif max(float(np.max(np.abs(d.qpos))),peak_speed)>1e-10 or switches!=0:
        raise AssertionError("idle switched joint drift")
    return {"steps":STEPS,"q_final_rad":float(d.qpos[0]),
            "switches":switches,"max_global_residual_m":peak_residual,
            "peak_abs_eq_force_N":peak_force,"peak_speed":peak_speed,
            "peak_step_q_rad":peak_delta_q,"peak_step_s_m":peak_delta_s,
            "force_energy_equivalence":"NOT_TESTED"}


def main():
    mu,m=compiled()
    out={"experiment":"SIM-026C4B2D3A","native_mujoco":mu.__version__,
         "original_source_data":"NONE",
         "synthetic_continuity":check_continuity(),
         "single_global_quartic_limit":global_quartic_failure(),
         "native_static":static_native(mu,m),
         "native_idle":native_dynamics(mu,m,False),
         "native_crossing":native_dynamics(mu,m,True),
         "original_anatomical_knee_dynamics":"NO_GO",
         "human_orthosis_clinical":"NO_GO"}
    print(json.dumps(out,sort_keys=True))
    return out


if __name__=="__main__":
    main()
