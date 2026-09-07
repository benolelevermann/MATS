from __future__ import annotations

import inspect
import unittest
from unittest.mock import patch

import numpy as np
import torch
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from scipy import ndimage as ndi

from custom_trainers.skeleton_recall.nnUNetTrainerSkeletonRecallCells import (
    SemanticFluxAuxiliaryNetwork,
    SkeletonContextFluxTargetTransform,
    add_label_safe_spatial_augmentation,
    nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1x,
    nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAug,
    nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAugDebug50,
    nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAugFluxAux,
    nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAugFluxAuxDebug50,
)


class DummyTransforms:
    def __init__(self, transforms):
        self.transforms = transforms


class DummySegmenter(torch.nn.Module):
    def forward(self, data):
        return [
            torch.cat((data, data, data), dim=1),
            torch.cat((data[:, :, ::2, ::2],) * 3, dim=1),
        ]


class ConnectivityTrainerTests(unittest.TestCase):
    def test_label_safe_augmentation_disables_interpolating_warps(self) -> None:
        spatial = SpatialTransform(
            (16, 16),
            patch_center_dist_from_border=0,
            random_crop=False,
            p_rotation=0.2,
            rotation=(-0.5, 0.5),
            p_scaling=0.2,
            scaling=(0.7, 1.4),
        )
        transforms = DummyTransforms([spatial])

        result = add_label_safe_spatial_augmentation(transforms, (16, 16))

        self.assertIs(result, transforms)
        self.assertEqual(spatial.p_rotation, 0.0)
        self.assertEqual(spatial.p_scaling, 0.0)
        self.assertEqual(len(transforms.transforms), 2)

    def test_label_safe_augmentation_preserves_one_pixel_line_connectivity(self) -> None:
        spatial = SpatialTransform(
            (16, 16),
            patch_center_dist_from_border=0,
            random_crop=False,
            p_rotation=0.2,
            rotation=(-np.pi, np.pi),
            p_scaling=0.2,
            scaling=(0.7, 1.4),
        )
        transforms = add_label_safe_spatial_augmentation(DummyTransforms([spatial]), (16, 16))
        segmentation = torch.zeros((1, 16, 16), dtype=torch.int16)
        segmentation[0, 8, 2:14] = 1

        for seed in range(20):
            np.random.seed(seed)
            torch.manual_seed(seed)
            transformed = transforms.transforms[0](
                image=torch.zeros((1, 16, 16)), segmentation=segmentation.clone()
            )
            transformed = transforms.transforms[1](**transformed)
            skeleton = transformed["segmentation"][0].numpy() == 1
            self.assertEqual(int(skeleton.sum()), 12)
            self.assertEqual(ndi.label(skeleton, structure=np.ones((3, 3)))[1], 1)

    def test_flux_target_has_unit_vectors_only_in_radius_seven_context(self) -> None:
        segmentation = torch.zeros((1, 21, 21), dtype=torch.int16)
        segmentation[0, 10, 5:16] = 1
        transform = SkeletonContextFluxTargetTransform(radius=7)

        result = transform.apply({"segmentation": segmentation})

        flux = result["flux"].numpy()
        mask = result["flux_mask"].numpy()[0].astype(bool)
        self.assertEqual(flux.shape, (2, 21, 21))
        self.assertTrue(mask[4, 10])
        self.assertFalse(mask[2, 10])
        norms = np.linalg.norm(flux[:, mask], axis=0)
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-5)

    def test_flux_wrapper_preserves_eval_inference_interface(self) -> None:
        network = SemanticFluxAuxiliaryNetwork(DummySegmenter(), 3, 2)
        data = torch.ones((2, 1, 16, 16))

        network.train()
        semantic, flux = network(data)
        self.assertIsInstance(semantic, list)
        self.assertEqual(flux.shape, (2, 2, 16, 16))

        network.eval()
        output = network(data)
        self.assertIsInstance(output, list)
        self.assertEqual(output[0].shape, (2, 3, 16, 16))

    def test_r4_and_r5_keep_r0_class_weighting(self) -> None:
        self.assertTrue(
            issubclass(
                nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAug,
                nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1x,
            )
        )
        self.assertEqual(
            nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAug.skeleton_recall_class_weights,
            (2.0, 1.0),
        )
        self.assertTrue(
            issubclass(
                nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAugFluxAux,
                nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAug,
            )
        )

    def test_debug_variants_keep_explicit_signature_and_50_epochs(self) -> None:
        pairs = (
            (
                nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAug,
                nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAugDebug50,
            ),
            (
                nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAugFluxAux,
                nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAugFluxAuxDebug50,
            ),
        )
        for parent, debug in pairs:
            with self.subTest(debug=debug.__name__):
                with patch.object(
                    parent,
                    "__init__",
                    lambda instance, *args, **kwargs: setattr(instance, "num_epochs", 1000),
                ):
                    trainer = debug({}, "2d", 0, {}, None)
                self.assertEqual(trainer.num_epochs, 50)
                self.assertEqual(
                    list(inspect.signature(debug.__init__).parameters),
                    ["self", "plans", "configuration", "fold", "dataset_json", "device"],
                )


if __name__ == "__main__":
    unittest.main()
