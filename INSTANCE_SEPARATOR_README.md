# Soma-seeded cell separation

This experiment changes only cell separation. Network 141 stays frozen and
continues to generate the semantic classes background, skeleton and soma.

For every detected soma, the separator receives four channels:

1. normalized microscopy image,
2. network-141 skeleton mask after hysteresis,
3. network-141 soma mask,
4. a binary mask highlighting exactly one queried soma.

Its output is the probability that each foreground pixel belongs to the
queried soma. The model is applied once per soma in a connected conflict group.
The highest probability becomes the primary local cell ID. A second cell ID is
retained where two soma queries both predict membership. Numeric IDs are only
selectors and are never learned as semantic classes.

## Data

Training uses only the instance maps of the 863 synthetic multi-cell scenes in
`nnUNet_raw/Dataset143_dataset141_plus_synthetic_multicell`. One scene produces
one query sample per cell: 2,339 training and 260 validation queries. Exact
90-degree rotations and flips preserve the one-pixel skeleton labels.

## Training

Five-epoch technical test:

```powershell
.\run_training_instance_separator.ps1
```

80-epoch run after manual acceptance of the technical test:

```powershell
.\run_training_instance_separator.ps1 -FullRun
```

Add `-Continue` to continue the corresponding run. Checkpoints and a CSV with
loss, membership Dice, skeleton recall and wrong-skeleton rate are written to
`instance_separator_results/` and are intentionally excluded from Git.

## Application and safety gate

`apply_instance_separator.py` expects a raw TIFF, the fixed semantic TIFF from
network 141 plus hysteresis, and a separator checkpoint. Single-soma components
are passed through directly; only components containing multiple somata are
sent through the learned separator.

The export is conservative. A whole multi-cell group is excluded from `cells/`
when any member contains at least eight disconnected or low-confidence
skeleton pixels. Rejected groups remain visible in the HTML gallery so they can
be diagnosed rather than silently accepted.

The five-epoch checkpoint is a feasibility test, not the final pipeline model.
