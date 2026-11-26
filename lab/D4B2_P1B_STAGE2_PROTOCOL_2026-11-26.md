# D4B2-P1b frozen protocol — stage-2 muscle-path fitting versus the stage-1 baseline

**Registered before first stage-2 public-source experiment, 2026-11-26.** This is an accuracy *comparison and screening* experiment, not final STS or muscle-force GO.

Previous real stage-1 source-native study: GitHub Actions run 45848997334 (9 source-free tests PASS), original public OpenSim RajagopalLaiUhlrich2023 (80 muscles), 4 core bilateral knee/hip poses plus 2 2023-specific >120° poses. Stage-1 2023 core worst absolute muscle-path length error 203.95778123 mm; worst scored knee/hip pair moment-arm RMS 57.8219192 mm; high-flexion worst length error 9.44103028 mm, worst moment-arm RMS 14.86977141 mm.

## Falsifiable P1b question

After converting the **same pinned publicly published 2023 model** with **the same pinned publicly available converter** through **stage 2** (`o2m convert --steps 1 2`), does the fitted `_cvt2.xml` reduce the error of 80 muscle-path lengths and bilateral knee/hip moment arms relative to the `_cvt1.xml` from *the same conversion job*?

## Immutable inputs and comparison

- Official original model `opensim-org/opensim-models@d9b05d470b1a481c222372c85b75772faf8f7792`, original 2023 model Git blob `d296dff62a1340a895dfe3a9ae8be03895b2e9f1`, 922865 bytes.
- Public converter `wafisharif/osim2mujoco@8c4d97e96f43c0d1676b8d58b0cff95af72a0f0f`, Python 3.12, OpenSim 4.6, MuJoCo 3.15.0, NumPy 2.5.3, `--steps 1 2`, **no speedy flag, no muscles removed**. Preserve both `cvt1` and `cvt2` transient outputs.
- Reuse **exact code `lab/d4b2_muscle_path_screen.py`** for source-vs-native comparisons, with original source constraint couplers and native joint equalities resolved in both models.
- Same 4 predeclared core poses and separately reported 2 high-flexion poses; 80 unique source muscles, 80 matching tendon/actuator names, all core and >120° path lengths, 4 knee/hip independent-coordinate path-length derivatives `-dL/dq` with h=1e−6 and 10h consistency check.
- Unchanged numerical/coverage validity gates: all 80 named muscles, no nonfinite values, source QErr≤1e−9 and native equality residual≤1e−9, consistent nonzero solver equality rows, ≥99% FD h vs 10h self-consistency on both engines. Four core + two highflex pose counts exact. **Revalidate the stage-1 same-run baseline** against the earlier run within 0.01 mm on worst absolute length error for core and high-flexion; otherwise do not compare or declare improvement.
- Publish aggregate lengths RMS mean/median/p95/worst, worst absolute path errors, scored moment-arm RMS mean/median/worst, count within 2 mm / within 5% source peak, source/native FD consistency and **signed improvements** `stage2 − stage1`. No muscle names, coefficients, arrays, XML, meshes or private patient data in logs/artifacts/history.
- Classification `SCREEN_VALID_MEASURED` means scientific measurement was valid. Whether fitted model **improved or worsened** should be reported explicitly for each metric; **no absolute path accuracy PASS** is registered here and no threshold is retrospectively optimized. A major worsening is a real result, not a workflow fix.

P1b compares skeletal constraints and static *musculotendon geometry*. Only after observing fitted path errors can a separate P1c force test (full `cvt3` stage, OpenSim static muscle equilibrium at activations 0, .25, .5, 1, passive tendon caveats) be designed. No dynamic sit-to-stand or human-clinical inference.
