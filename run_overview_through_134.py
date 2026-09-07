from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tifffile


def find_one(root: Path, pattern: str) -> Path:
    matches = sorted(p for p in root.glob(pattern) if p.is_dir())
    if len(matches) != 1:
        raise RuntimeError(
            f'Expected exactly one match for {pattern} in {root}, found:\n'
            + '\n'.join(str(p) for p in matches)
        )
    return matches[0]


def find_model_folder(results_root: Path, fold: int) -> Path:
    dataset_dir = find_one(results_root, 'Dataset134_*')
    matches = []
    for model_dir in dataset_dir.glob('*__*__2d'):
        if (model_dir / f'fold_{fold}' / 'checkpoint_best.pth').exists():
            matches.append(model_dir)
    if len(matches) != 1:
        raise RuntimeError(
            'Dataset134 model folder with checkpoint_best.pth was not found uniquely:\n'
            + '\n'.join(str(p) for p in matches)
        )
    return matches[0]


def load_original(path: Path, projection_axis: int) -> np.ndarray:
    image = np.squeeze(np.asarray(tifffile.imread(path)))
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        print(f'Original stack shape: {image.shape}')
        print(f'Creating max projection along axis {projection_axis}.')
        return np.max(image, axis=projection_axis)
    raise RuntimeError(f'Original must be 2D or 3D, found {image.shape}.')


def load_probabilities(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    with np.load(path) as archive:
        if 'probabilities' in archive:
            probabilities = np.asarray(archive['probabilities'])
        elif 'softmax' in archive:
            probabilities = np.asarray(archive['softmax'])
        elif len(archive.files) == 1:
            probabilities = np.asarray(archive[archive.files[0]])
        else:
            raise RuntimeError(f'Unknown NPZ content in {path.name}: {archive.files}')

    probabilities = np.squeeze(probabilities)
    if probabilities.ndim != 3:
        raise RuntimeError(f'Expected 3D probabilities, found {probabilities.shape}.')

    if probabilities.shape[1:] == expected_shape:
        chw = probabilities
    elif probabilities.shape[:2] == expected_shape:
        chw = np.moveaxis(probabilities, -1, 0)
    else:
        raise RuntimeError(
            f'Probability shape {probabilities.shape} does not match image {expected_shape}.'
        )

    if chw.shape[0] < 3:
        raise RuntimeError(f'Expected three classes, found {chw.shape[0]}.')
    return chw.astype(np.float32, copy=False)


def colorize(label: np.ndarray) -> np.ndarray:
    label = np.squeeze(label)
    rgb = np.zeros((*label.shape, 3), dtype=np.float32)
    rgb[label == 1] = (0.0, 1.0, 0.0)
    rgb[label == 2] = (1.0, 0.0, 1.0)
    return rgb


def display_normalize(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image)
    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return np.zeros_like(image)
    return np.clip((image - low) / (high - low), 0, 1)


def run_prediction(project: Path, input_dir: Path, output_dir: Path, device_name: str, fold: int) -> None:
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    if device_name == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but not available.')
        device = torch.device('cuda', 0)
        on_device = True
    else:
        device = torch.device('cpu')
        on_device = False

    model_folder = find_model_folder(project / 'nnUNet_results', fold)
    print(f'Loading model: {model_folder}')

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=on_device,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_folder),
        use_folds=(fold,),
        checkpoint_name='checkpoint_best.pth',
    )
    predictor.predict_from_files(
        str(input_dir),
        str(output_dir),
        save_probabilities=True,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(r'C:\Ole\20260721_CellClassification_v2'))
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--net129-output-dir', type=Path, required=True)
    parser.add_argument('--case-id', default='overview')
    parser.add_argument('--output-root', type=Path, default=None)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--projection-axis', type=int, default=0)
    args = parser.parse_args()

    project = args.project.resolve()
    output_root = args.output_root.resolve() if args.output_root else project / 'overview_134_test'
    input_dir = output_root / 'input'
    prediction_dir = output_root / 'prediction'
    input_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    os.environ['nnUNet_raw'] = str(project / 'nnUNet_raw')
    os.environ['nnUNet_preprocessed'] = str(project / 'nnUNet_preprocessed')
    os.environ['nnUNet_results'] = str(project / 'nnUNet_results')

    original = load_original(args.original.resolve(), args.projection_axis)
    npz_path = args.net129_output_dir.resolve() / f'{args.case_id}.npz'
    if not npz_path.exists():
        npz_files = sorted(args.net129_output_dir.resolve().glob('*.npz'))
        if len(npz_files) == 1:
            npz_path = npz_files[0]
        else:
            raise FileNotFoundError(npz_path)

    probabilities = load_probabilities(npz_path, original.shape)
    p_skeleton = np.clip(probabilities[1], 0, 1).astype(np.float32)
    p_soma = np.clip(probabilities[2], 0, 1).astype(np.float32)

    tifffile.imwrite(input_dir / f'{args.case_id}_0000.tif', original, photometric='minisblack')
    tifffile.imwrite(input_dir / f'{args.case_id}_0001.tif', p_skeleton, photometric='minisblack')
    tifffile.imwrite(input_dir / f'{args.case_id}_0002.tif', p_soma, photometric='minisblack')

    run_prediction(project, input_dir, prediction_dir, args.device, args.fold)

    prediction134_path = prediction_dir / f'{args.case_id}.tif'
    prediction134 = np.squeeze(tifffile.imread(prediction134_path)).astype(np.uint8)

    hard129_path = args.net129_output_dir.resolve() / f'{args.case_id}.tif'
    if hard129_path.exists():
        hard129 = np.squeeze(tifffile.imread(hard129_path)).astype(np.uint8)
    else:
        background = np.clip(1 - p_skeleton - p_soma, 0, 1)
        hard129 = np.argmax(np.stack([background, p_skeleton, p_soma]), axis=0).astype(np.uint8)

    changed = hard129 != prediction134
    tifffile.imwrite(output_root / 'changed_pixels_129_to_134.tif', changed.astype(np.uint8), photometric='minisblack')

    figure, axes = plt.subplots(1, 6, figsize=(24, 5))
    panels = [
        (display_normalize(original), 'Original / max projection', 'gray'),
        (p_skeleton, 'Input 134: P(Skeleton)', 'gray'),
        (p_soma, 'Input 134: P(Soma)', 'gray'),
        (colorize(hard129), 'Output network 129', None),
        (colorize(prediction134), 'Output network 134', None),
        (changed.astype(np.uint8), 'Changed pixels', 'gray'),
    ]
    for axis, (image, title, cmap) in zip(axes, panels):
        if cmap is None:
            axis.imshow(image)
        else:
            axis.imshow(image, cmap=cmap, vmin=0, vmax=1)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.tight_layout()
    comparison_path = output_root / 'overview_129_vs_134.png'
    figure.savefig(comparison_path, dpi=180, bbox_inches='tight')
    plt.close(figure)

    print('\nOverview test finished')
    print(f'Network 134 output: {prediction134_path}')
    print(f'Comparison image:   {comparison_path}')
    print(f'Changed pixels:     {int(changed.sum())}')


if __name__ == '__main__':
    main()
