# Dataset137: Dataset136 plus selected div10 tracings

[Zur Dokumentationsübersicht](../README.md)

This three-stage build deliberately leaves both the `X:` tracing source and
`Dataset136` unchanged.

## What is selected?

Each cell's `source_fov.txt` is joined as text to `Encryption` in
`toTrace_EncryptionKey.csv`. The treatment is read from the final token of
`OriginalName`.

Only exact treatments `Blebbistatin` and `DMSO` are excluded. `dilutedDMSO`
is retained because it is a distinct final treatment token.

## Files

1. `select_div10_tracings_for_dataset137.py`
   creates an auditable manifest only.
2. `rasterize_div10_training_masks_fiji.py`
   runs in Fiji and converts `seg.swc` plus `soma.zip` to a 0/1/2 label TIFF.
3. `build_dataset137_from_dataset136_and_div10.py`
   copies the old 2,013 Dataset136 cases plus verified new pairs to Dataset137.

The SWC reader uses its own header's voxel calibration. For the demonstrated
cells this is `0.406249892 µm/pixel`, so an SWC coordinate is converted as
`pixel = coordinate_µm / 0.406249892`.

## Safety properties

- A non-empty output directory is rejected.
- Missing, unmatched, duplicate, or ambiguous encryption-key mappings are
  not included.
- A new label must be 2-D, match raw-image geometry, only contain 0/1/2, and
  contain both skeleton and soma pixels.
- New IDs are namespaced (`div10_cell...`) and cannot collide with the numeric
  Dataset136 IDs.
- The build writes manifests and reports for later audit.

## Do not copy Dataset136's split

Dataset137 must receive a fresh preprocessing and fold split, otherwise the
new cases will not participate correctly in training/validation.
