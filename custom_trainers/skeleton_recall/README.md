# Skeleton Recall trainer for Dataset136

This external trainer is compatible with the locally inspected environment:

- nnU-Net 2.8.1
- Python 3.14
- PyTorch 2.13
- 2D multiclass labels: background=0, skeleton=1, soma=2

The default trainer uses the paper's full additional weight:

`Dice + Cross-Entropy + 1.0 * Skeleton Recall`

The optional `nnUNetTrainerSkeletonRecallCellsW01` uses the conservative paper
setting `0.1`.

The original Dataset136, preprocessing and baseline checkpoints are not changed.

## One-pixel full-resolution experiment

`nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xFullResOnly` keeps the
existing Skeleton-2x/Soma-1x loss but assigns deep-supervision weights
`[1, 0, 0, ...]`. Network outputs and targets are still generated at all
configured scales; only the full-resolution output contributes a gradient.

`nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xFullResOnlyDebug50` is the
same experiment limited to 50 epochs. Its distinct class name gives it a
separate nnU-Net result folder, so it cannot overwrite the 1000-epoch run.

Load these source-controlled external trainers with:

```powershell
$env:nnUNet_extTrainer = "$PWD\custom_trainers\skeleton_recall"
```

## Connectivity experiments R4 and R5

`nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAug` keeps the R0
architecture, loss and deep supervision. It disables interpolating arbitrary
rotation/scaling and inserts exact 90-degree rotation while retaining mirroring
and all intensity augmentations. The `...Debug50` class is its smoke test.

`nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAugFluxAux` adds a
training-only two-channel context-flux head (radius 7, weight 0.1). In eval mode
the wrapper returns the ordinary semantic outputs, so nnU-Net validation and
prediction remain compatible. Its `...Debug50` class is the smoke test.
