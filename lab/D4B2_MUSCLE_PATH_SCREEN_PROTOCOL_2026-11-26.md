# D4B2-P1a — original-public-model muscle path screening protocol

**Frozen before first D4B2 public-source muscle test (2026-11-26).** This is a new, limited *screen*, not a claim of full musculotendon/dynamic fidelity. The D4B1 native anatomical FK experiment was run successfully beforehand in public GitHub Actions #45633985578. A D4B1 kinematics PASS does not automatically imply muscle-path validity.

## P1a question

At static source-native corresponding coordinate states, does the converted *step-1* MuJoCo model retain all 80 OpenSim musculotendon path lengths and their **independent-coordinate path-length derivatives**? What is the error distribution, including 2023-specific high knee flexion?

## Inputs and stages

- Public upstream `opensim-org/opensim-models@d9b05d470b1a481c222372c85b75772faf8f7792`: Rajagopal2016 and RajagopalLaiUhlrich2023 (each original model SHA-checked). The published source files are checked out transiently only.
- Public converter `wafisharif/osim2mujoco@8c4d97e96f43c0d1676b8d58b0cff95af72a0f0f`, **stage 1 only** (`o2m convert --steps 1`); OpenSim 4.6, MuJoCo 3.15.0, Python 3.12.
- **No stage 2 muscle-path fitting and no stage 3 force fitting yet.** The step-1 path result establishes the pre-optimization baseline; it may be inaccurate and must not be rebranded as the finished muscle model.
- Core: 4 predeclared bilateral knee/hip poses on both models; 2023-only extension: 2 predeclared bilateral >120° knee poses. All other source coordinates held to original model defaults; enforced OpenSim coordinate couplers applied directly, MuJoCo named joint equalities solved through converter's pinned helper.
- All **80 named source muscles** must map one-to-one to MuJoCo `<muscle>_tendon` and corresponding muscle actuator. All 80 × 4 source/native path-length comparisons per core model and 80 × 2 extra on 2023 are retained transiently, but only aggregate errors are published.
- Per pose, for independent `knee_angle_r, knee_angle_l, hip_flexion_r, hip_flexion_l`, calculate path moment arm **−dL/dq** by symmetric finite difference at **h = 1e−6 rad** on each simulator, *re-solving any source/native coordinate coupling at both ±h*. Recheck finite differences with 10h; this is the same physical path-length derivative definition on both platforms. This is intentionally not `computeMomentArm` because source wrapped muscles can disagree with it (see upstream Station 3 methodology).
- Score muscle-coordinate pairs where maximum OpenSim moment-arm magnitude in sampled positions ≥**1 mm**; report non-spanning/near-zero pairs separately. Report mean/median/max error in mm, RMS and percent-of-peak for scored moment arm pairs, per-muscle path-length RMS and 95th percentile, plus overall worst observed absolute length error. Report FD 1e−6 vs 1e−5 self-consistency coverage on source and native.

## Fixed validity gates (not tuned after output)

- Two source blobs match fixed upstream Git SHA and byte size; 22 physical bodies, 22 OpenSim 4.6 runtime joints, 39 coordinates, 80 unique muscles per original model, 75.337 kg total body mass.
- Every source muscle maps to exactly one native named tendon and matching actuator (native muscle activation dynamics); no silent omissions.
- All sampled lengths, FD-derived moment arms, and residuals finite, nonnegative path lengths, source QErr ≤1e−9 and native equality residual ≤1e−9, *every* original and perturbed coordinate state. Every state has a nonzero, stable active equality-row count.
- FD internal repeatability: **≥99%** of raw source and native muscle-coordinate derivative samples differ by ≤1e−6 m between h and 10h; source discontinuities or numerical instability must be reported as `FD_INCONSISTENT`, not made into a path-accuracy PASS. None of these 99%/1e−6 gates controls the absolute accuracy of muscle paths.
- The P1a result may only be reported as `SCREEN_VALID_MEASURED` (when all validity gates pass) or `SCREEN_INVALID` / `ERROR`. **No PASS/FAIL of muscle *accuracy***, since an absolute allowable RMS path/force error was not preregistered for this distinct model/application. A measured high error is substantive evidence to investigate, not grounds for retuning to a favorable threshold.
- Every run must report count and error aggregates **for core and 2023 high-flexion separately**, not hide worst poses by pooling. No derived XML, mesh, per-pose vectors or force arrays in GitHub history, Actions caches or downloadable artifacts.

## Next gating

If P1a source-native step-1 path diagnostics are reproducible, compare them to a *stage-2 fitted* converter result on the same poses. Force-response validation requires stage-3 fitting and comparisons at activations 0, .25, .5 and 1, plus explicit OpenSim muscle-equilibrium and tendon compliance diagnostics. These are **separate experiments**, and a purely static muscle check never proves a reliable dynamic sit-to-stand assistance design.

**Ethical/clinical scope:** no patient input, diagnosis, medical-device efficacy, human trial, or wearer/hardware operation is performed or implied.
