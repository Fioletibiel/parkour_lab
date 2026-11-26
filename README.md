# Parkour Lab — public computational sandbox

This is a public, **source-separated** test environment for computational experiments.
Approved private research records and finished implementations remain in the private `Parkour` repository.
Public GitHub Actions use standard free runners; public workflow logs are visible to everyone.

## Strict publication boundary
- The repository's **tracked files** must contain only authored experiment code, synthetic fixtures, and references to independently public third-party sources. Never commit extracted original anatomical model files, original/converted .osim/.xml/.mjcf, mesh geometry, derived source arrays or participant data.
- An explicitly scoped workflow may temporarily check out an **already publicly published** OpenSim example model and an open-source converter on the public CI runner *only to measure reproducible aggregate results*. These ephemeral sources and converted models are never committed, uploaded as artifacts, cached or pasted into logs.
- Do not check out the private Parkour repo or read any private files, tokens, secrets, credentials, patient data, unpublished clinical datasets or private source-specific transforms into this public job. Restrict permissions to `contents: read`.
- No paid runners. No cloud resource creation. Public logs must contain only aggregate errors/statuses; never detailed anatomical source data.
- A successful public-source OpenSim kinematics result does not establish muscle-force fidelity, dynamic sit-to-stand, assistance feasibility, or clinical safety.
- All research interpretation and transfer of reviewed aggregate observations into the authoritative private project remain separate.

## D4B1: public-source native OpenSim–MuJoCo probe (experimental branch)

`experiment/d4b1-public-open-model-native-20261125` contains `lab/d4b1_open_model_native.py` and its GitHub Actions workflow. Both inputs are retrieved transiently from their **original public** repositories at fixed source commits:

- `opensim-org/opensim-models@d9b05d470b1a481c222372c85b75772faf8f7792`, Rajagopal2016 and RajagopalLaiUhlrich2023. Original published scientific work and citations are identified in `Models/Rajagopal/README.txt` upstream.
- `wafisharif/osim2mujoco@8c4d97e96f43c0d1676b8d58b0cff95af72a0f0f` (converter, upstream Apache-2.0 licensing), Python 3.12, OpenSim 4.6 and MuJoCo 3.15.0.

The probe compares 4 synthetic **coordinate inputs** for both openly published source models and a distinct 2-pose >120° high-flexion extension for the 2023 model. Comparison uses actual OpenSim source and actual MuJoCo native FK; the converter's own joint dependency helper is used, so this is **not a completely independent model conversion algorithm**. It records only counts, max errors and PASS/FAIL in public stdout. No participant data or private original-source assets are involved.

This experimental workflow must be evaluated on its own merits; do not treat successful CI startup or the older synthetic results as native anatomical validation. For detailed governance, refer to the private project documentation (not copied here).

## D4B2-P1a — real public-source 80-muscle path screening

The **2026-11-26** native OpenSim-to-MuJoCo muscle-path baseline ran on the public standard runner as GitHub Actions [45848997334](https://github.com/Fioletibiel/parkour_lab/actions/runs/45848997334), using **only transient, publicly published official Rajagopal source models**. 9/9 source-free preflight tests passed; all 80 named muscles in each model were compared against unoptimized stage-1 MuJoCo tendons. Four core static poses per original model and two 2023-specific >120° poses were included. Source/native finite difference self-consistency, original QErr and native equality residual gates passed.

**Important scientific caution:** the measured stage-1 paths still exhibit substantial **outlier errors**. The 2016 and 2023 core worst absolute muscle-path length errors were approximately **204 mm**, while the worst scored knee/hip arm pair RMS errors were approximately **57.8 mm**. The 2023 beyond-120° pose extension showed ~**9.44 mm** worst path-length error and ~**14.9 mm** worst arm-pair RMS error. Small median muscle errors do **not** make this an accurate anatomical muscle model.

This study's `SCREEN_VALID_MEASURED` outcome refers **only to coverage and soundness of the measurements**. Stage-2 fitted muscle-path accuracy, stage-3 muscle forces, tendon elasticity, dynamic sit-to-stand and clinical safety were **not** validated. See the [frozen D4B2-P1a protocol](lab/D4B2_MUSCLE_PATH_SCREEN_PROTOCOL_2026-11-26.md). Detailed interpretation and follow-up are maintained in the private research repository.

## Original source-free synthetic engine experiments

The following rules describe the legacy **synthetic-only** jobs, not the separate D4B1 public-source native workflow above. All results require review before use in private research.

## Legacy synthetic experiment boundaries
- Track only wholly synthetic fixtures, source-free mathematical examples and authored reproducible scripts. The separate D4B1 workflow may temporarily read public official OpenSim examples but must never add them to tracked files or artifacts.
- **Never commit** private repository files, original/derived model XML/MJCF, anatomical parameter arrays, patient/participant data, proprietary test sets, unpublished reports, tokens, API keys, or private source-specific transforms.
- Public GitHub Actions **logs and artifacts are public**. Print aggregate synthetic measurements in synthetic workflows and only aggregate error/status measurements in public-source native workflows. Restrict workflows to least-privilege `contents: read`; no secrets or private checkouts.
- Use **standard** `ubuntu-24.04` GitHub-hosted runner. Avoid larger paid runners and persistent caches/artifacts unless deliberately requested.
- Do not regard synthetic PASS as anatomical dynamic validity, human assistance, medical or clinical safety, or original-source knee equivalence.
- All scientific decisions, confidential source checks and integration remain outside this public lab.

## SIM-026C4B2D2 synthetic engine test
`lab/sim026c4b2d2.py` builds a fresh two-body chain and applies one scalar native quartic equality constraint. Its seven frozen angle samples (rad), 500-step idle and 500-step torque-driven paths, and unit/mutation tests are synthetic, **not derived from any original model**.

Run locally:
```sh
python -m pip install --only-binary=:all: 'mujoco==3.3.6'
python -m unittest discover -s tests -v
python -m lab.sim026c4b2d2
```

GitHub Actions executes both; review the run/job logs before exporting a compact verification result. A successful run here is evidence about only a minimal synthetic connected mechanism—not about any real skeleton.

No rights to redistribute third-party anatomical model files or derivatives are implied by this repository. Public-source native processing is ephemeral and outputs only aggregate numeric validation statistics.
