# Permanent evaluation hold-outs

`div10_cc.csv` identifies the 489 manually traced cells stored locally in
`EvoTest/div10_CC`. They are evaluation data from 2026-09-10 onward and must
not be copied, transformed, used as synthetic donors, or otherwise used for
training, validation-based checkpoint selection, or hyperparameter tuning.

Before a dataset is preprocessed or trained, run:

```powershell
.\.venv\Scripts\python.exe .\evo_test_holdout.py check `
  --dataset .\nnUNet_raw\DatasetNNN_name
```

The maintained training entry points call this check automatically. The check
uses SHA-256 image content and the unique `evotest_div10cc_*` IDs; generic cell
numbers alone are not treated as evidence because other exports can reuse them.

The TIFF/SWC/ROI test data are deliberately ignored by Git. Back them up as a
directory; the small tracked CSV is the durable identity registry, not a copy
of the microscopy data.
