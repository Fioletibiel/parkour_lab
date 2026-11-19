"""D3B synthetic hybrid equality: independent reduced Lagrange dynamics and energy checks.

Public original-authored 2-body toy only. No anatomical model, private data, subjects
or original source spline arrays. Frozen scientific criteria in private prereg.
"""
from __future__ import annotations
import json
import math
import numpy as np

from lab import sim026c4b2d3a as fixture

ENGINE="3.3.6"
TIMESTEPS=(.002,.001,.0005)
HORIZON=.16
PULSE_CHANGE=.08
INERTIA_Z=.02+.02
M_PARENT=1.1
M_CHILD=.6
X_PARENT=.1
X_CHILD=1.2
DAMP=.2
SCENARIOS=(
    ("FORWARD",.68,.8,0.),
    ("REVERSE",.72,-.8,0.),
    ("PULSE",.68,.8,0.),
    ("GRAVITY",.68,.8,1.),
)
MAX_Q_ERROR=.008
MAX_V_ERROR=.04
MAX_CONSTRAINT=.0002
MAX_ENERGY_NET=.004
MAX_STEP_DEFECT=.0015
MAX_EQ_FORCE=10.
MAX_EQ_FORCE_CHANGE=3.


def f(q,derivative=0):
    return fixture.target(q,derivative)


def energy_mass(q):
    a=X_CHILD+f(q)
    p=f(q,1)
    return INERTIA_Z+M_PARENT*X_PARENT**2+M_CHILD*(a*a+p*p)


def energy_mass_derivative(q):
    a=X_CHILD+f(q)
    p=f(q,1)
    pp=f(q,2)
    return 2*M_CHILD*(a*p+p*pp)


def potential(q,g):
    return g*(M_PARENT*X_PARENT+M_CHILD*(X_CHILD+f(q)))*math.sin(q)


def potential_gradient(q,g):
    return g*((M_PARENT*X_PARENT+M_CHILD*(X_CHILD+f(q)))*math.cos(q)
              +M_CHILD*f(q,1)*math.sin(q))


def torque_at(name,t):
    if name=="FORWARD":return .1
    if name=="REVERSE":return -.1
    if name=="PULSE":return .12 if t<PULSE_CHANGE-1e-12 else -.04
    if name=="GRAVITY":return 0.
    raise ValueError("unexpected source-free scenario")


def reduced_acceleration(q,v,tau,g):
    mass=energy_mass(q)
    if mass<=0 or not math.isfinite(mass):
        raise ValueError("nonpositive synthetic constrained mass")
    b=DAMP*(1+f(q,1)**2)
    return (tau-.5*energy_mass_derivative(q)*v*v-b*v-potential_gradient(q,g))/mass


def reference_step(y,t,dt,scenario,g):
    """Independent reduced analytic 1-DoF RK4, no native simulator calls."""
    def deriv(v,tt):
        return np.array((v[1],reduced_acceleration(float(v[0]),float(v[1]),
                                                    torque_at(scenario,tt),g)))
    k1=deriv(y,t)
    k2=deriv(y+.5*dt*k1,t+.5*dt)
    k3=deriv(y+.5*dt*k2,t+.5*dt)
    k4=deriv(y+dt*k3,t+dt)
    return y+(dt/6)*(k1+2*k2+2*k3+k4)


def native_energy(q,s,vq,vs,g):
    """Actual 2-DoF kinetic energy; do not project s/velocity onto constraint."""
    kinetic=.5*(INERTIA_Z+M_PARENT*X_PARENT**2+M_CHILD*(X_CHILD+s)**2)*vq*vq
    kinetic+=.5*M_CHILD*vs*vs
    V=g*(M_PARENT*X_PARENT+M_CHILD*(X_CHILD+s))*math.sin(q)
    return kinetic+V


def native_power(vq,vs,tau):
    return tau*vq-DAMP*(vq*vq+vs*vs)


def run_case(name,q0,v0,g,dt):
    if name not in [v[0] for v in SCENARIOS] or dt not in TIMESTEPS:
        raise ValueError("unfrozen test variant")
    mu,model=fixture.compiled()
    model.opt.timestep=dt
    model.opt.gravity[:]=[0.,-g,0.]
    data=mu.MjData(model)
    # Initialization is the ONLY place joint coordinates/velocities are assigned.
    data.qpos[:]=[q0,f(q0)]
    data.qvel[:]=[v0,f(q0,1)*v0]
    fixture.select_active(data,q0)
    mu.mj_forward(model,data)
    ref=np.array((q0,v0),dtype=float)
    prev_e=native_energy(*[float(x) for x in data.qpos],
                         *[float(x) for x in data.qvel],g)
    e0=prev_e
    prev_power=native_power(float(data.qvel[0]),float(data.qvel[1]),
                            torque_at(name,0))
    n=round(HORIZON/dt)
    if abs(n*dt-HORIZON)>1e-14:
        raise ValueError("non-frozen physical horizon")
    switches=0
    switch_dforce=[]
    peak_delta_force=peak_abs_force=peak_defect=peak_res=0.
    total_work_damping=0.
    previous_force=None
    peak_speed=0.
    for index in range(n):
        t=index*dt
        before_branch=int(data.eq_active[1])
        new_branch=fixture.select_active(data,float(data.qpos[0]))
        did_switch=bool(before_branch!=new_branch)
        switches+=int(did_switch)
        tau=torque_at(name,t)
        data.qfrc_applied[:]=[tau,0.]
        # Never reset qpos, qvel or native solver state after initialization.
        mu.mj_step(model,data)
        if any(w.number for w in data.warning):
            raise AssertionError("native warning in D3B")
        if not all(np.all(np.isfinite(item)) for item in (data.qpos,data.qvel,
                                                         data.efc_force,data.xpos)):
            raise AssertionError("native nonfinite D3B state")
        q,s=(float(v) for v in data.qpos)
        vq,vs=(float(v) for v in data.qvel)
        if not 0<=q<=1.6:
            raise AssertionError("synthetic native q domain escape")
        peak_speed=max(peak_speed,abs(vq),abs(vs))
        if peak_speed>=10:
            raise AssertionError("native joint speed threshold violated")
        residual=abs(s-f(q))
        peak_res=max(peak_res,residual)
        if peak_res>MAX_CONSTRAINT:
            raise AssertionError("native constraint geometry drift")
        force=float(data.efc_force[0]) if data.nefc==1 else float("nan")
        if not math.isfinite(force):
            raise AssertionError("native active equality missing")
        peak_abs_force=max(peak_abs_force,abs(force))
        if peak_abs_force>MAX_EQ_FORCE:
            raise AssertionError("native native equality force exceeded frozen bound")
        if previous_force is not None:
            change=abs(force-previous_force)
            peak_delta_force=max(peak_delta_force,change)
            if did_switch:
                switch_dforce.append(change)
            if change>MAX_EQ_FORCE_CHANGE:
                raise AssertionError("sampled equality force jump too large")
        previous_force=force
        ee=native_energy(q,s,vq,vs,g)
        p_after=native_power(vq,vs,tau)
        # On a pulse transition integrate separate constant effort over each step.
        supplied=.5*(native_power(float(data.qvel[0]),float(data.qvel[1]),tau)+prev_power)*dt
        # previous q/s velocities must come from the pre-step snapshot; assigned below.
        # We compute with the conserved, actual post state and prior actual values.
        peak_defect=max(peak_defect,abs(ee-prev_e-supplied))
        total_work_damping+=supplied
        prev_e=ee
        prev_power=p_after
        ref=reference_step(ref,t,dt,name,g)
    net=abs(prev_e-e0-total_work_damping)
    q_err=abs(float(data.qpos[0])-float(ref[0]))
    v_err=abs(float(data.qvel[0])-float(ref[1]))
    if switches<1:
        raise AssertionError("missing actual synthetic knot crossing")
    if q_err>MAX_Q_ERROR or v_err>MAX_V_ERROR:
        raise AssertionError("native vs independent Lagrange reference disagrees")
    if net>MAX_ENERGY_NET or peak_defect>MAX_STEP_DEFECT:
        raise AssertionError("native energy/work/damping mismatch")
    return {"scenario":name,"dt_s":dt,"steps":n,
            "q_native_rad":float(data.qpos[0]),
            "q_reference_rad":float(ref[0]),
            "qd_native":float(data.qvel[0]),
            "qd_reference":float(ref[1]),
            "end_q_error_rad":q_err,"end_qd_error":v_err,
            "switch_count":switches,"peak_constraint_residual_m":peak_res,
            "energy_balance_error_J":net,"peak_step_energy_defect_J":peak_defect,
            "peak_abs_constraint_force_N":peak_abs_force,
            "peak_sample_force_jump_N":peak_delta_force,
            "force_jumps_at_switch_N":switch_dforce,
            "peak_abs_joint_speed":peak_speed}


def run_all():
    if fixture.VERSION!=ENGINE:
        raise ValueError("wrong base synthetic source-free engine")
    result=[]
    for name,q0,v0,g in SCENARIOS:
        for dt in TIMESTEPS:
            result.append(run_case(name,q0,v0,g,dt))
        coarse,fine=result[-3],result[-1]
        if (fine["end_q_error_rad"]>coarse["end_q_error_rad"]+.001 or
            fine["end_qd_error"]>coarse["end_qd_error"]+.01):
            raise AssertionError("refinement at dt=.0005 grossly worse than dt=.002")
    return {"experiment":"SIM-026C4B2D3B",
            "engine":ENGINE,"public_synthetic_source_data_only":True,
            "scenarios":len(SCENARIOS),"timesteps":len(TIMESTEPS),
            "cases":len(result),
            "max_q_error_rad":max(x["end_q_error_rad"] for x in result),
            "max_qd_error":max(x["end_qd_error"] for x in result),
            "max_energy_balance_error_J":max(x["energy_balance_error_J"] for x in result),
            "max_step_defect_J":max(x["peak_step_energy_defect_J"] for x in result),
            "max_equality_residual_m":max(x["peak_constraint_residual_m"] for x in result),
            "max_sampled_force_jump_N":max(x["peak_sample_force_jump_N"] for x in result),
            "all_knot_switches":sum(x["switch_count"] for x in result),
            "sampled_force_jump_does_not_prove_pointwise_continuity":True,
            "original_source_knee_equivalence":"NO_GO",
            "clinical_human_orthosis":"NO_GO",
            "case_details":result}


def main():
    output=run_all()
    print(json.dumps(output,sort_keys=True))
    return output


if __name__=="__main__":
    main()
