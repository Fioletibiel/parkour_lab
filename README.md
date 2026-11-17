# Parkour Lab — public computational sandbox

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
