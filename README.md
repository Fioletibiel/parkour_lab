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

## Original source-free synthetic engine experiments

This is an **independent public testing environment** for **synthetic, non-proprietary** numerical and simulator experiments. Its results require review and independent promotion before use in any private research project.

## Scope and disclosure boundary
- Only wholly synthetic fixtures, source-free mathematical examples and reproducible scripts.
- **Never** upload private repository files, model XML/MJCF extracted from another source, original OpenSim Rajagopal2016 model and its parameter arrays, patient/participant data, proprietary test sets, unpublished reports, GitHub tokens, API keys, private repository links, or source-specific transforms.
- Public GitHub Actions **logs and artifacts are public**. Print aggregate synthetic measurements only. Restrict workflows to least-privilege `contents: read`; no secrets or private checkouts.
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

No third-party anatomical model, data or redistribution rights are implied by this repository.
