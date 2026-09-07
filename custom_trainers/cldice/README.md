# Experiment R6: clDice on Dataset138

This external nnU-Net trainer adds the topology-preserving soft-clDice term to
semantic class 1 (skeleton) only. The existing Dice plus cross-entropy loss and
its deep-supervision weights are unchanged. clDice uses alpha `0.1`, five
soft-skeleton iterations, and only the full-resolution network output. Samples
without skeleton ground truth are ignored by clDice but remain supervised by
Dice/CE.

Start the 50-epoch technical test from the project root:

```powershell
.\run_experiment06_cldice.ps1 -Device cuda
```

Resume it with `-Continue`. Start the separate 1000-epoch result with
`-FullRun`; add `-Continue` when resuming that run. CPU is supported through
`-Device cpu`, but is expected to be slow.

After validation, compare the raw 0/1/2 TIFF predictions with the production
baseline:

```powershell
.\.venv\Scripts\python.exe .\evaluate_cldice_validation.py
```

The evaluator writes `per_case.csv`, `summary.json`, and `comparison.md` under
`skeleton_connectivity_runs/experiment06_cldice_comparison`. For the full run,
pass its validation directory explicitly with `--candidate-dir`.

The strict score uses exact pixel overlap. The tolerant score dilates the
opposite support by a one-pixel Euclidean disk before calculating topology
precision and sensitivity, so a localization error is distinguished from a
real gap. Both scores are computed on raw validation masks, before hysteresis
or reconnection.
