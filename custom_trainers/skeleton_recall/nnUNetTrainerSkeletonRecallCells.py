from __future__ import annotations

import os
from typing import List, Tuple, Union

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
    NonDetMultiThreadedAugmenter,
)
from batchgenerators.dataloading.single_threaded_augmenter import (
    SingleThreadedAugmenter,
)
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.spatial.rot90 import Rot90Transform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import (
    DownsampleSegForDSTransform,
)
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import (
    MemoryEfficientSoftDiceLoss,
    get_tp_fp_fn_tn,
)
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.helpers import softmax_helper_dim1
from nnunetv2.utilities.plans_handling.plans_handler import (
    ConfigurationManager,
    PlansManager,
)
from scipy import ndimage as ndi
from skimage.morphology import diamond, dilation, skeletonize
from threadpoolctl import threadpool_limits
from torch import autocast, nn
from torch.nn import functional as F


def deep_supervision_loss_weights(
    number_of_scales: int,
    *,
    full_resolution_only: bool,
    tiny_last_weight: bool = False,
) -> np.ndarray:
    """Return normalized loss weights without changing the network outputs.

    The baseline branch deliberately reproduces nnU-Net's existing geometric
    weights. The experimental branch keeps only the full-resolution loss so a
    one-pixel skeleton is never supervised through a downsampled target.
    """

    if number_of_scales <= 0:
        raise ValueError("number_of_scales must be positive")
    if full_resolution_only:
        weights = np.zeros(number_of_scales, dtype=np.float64)
        weights[0] = 1.0
        return weights

    weights = np.array(
        [1 / (2**index) for index in range(number_of_scales)],
        dtype=np.float64,
    )
    weights[-1] = 1e-6 if tiny_last_weight else 0.0
    return weights / weights.sum()


class TubedSkeletonTargetTransform(BasicTransform):
    """Create a class-labelled, radius-N tube around the GT foreground skeleton."""

    def __init__(self, radius: int = 2):
        super().__init__()
        self.radius = int(radius)

    def apply(self, data_dict, **params):
        segmentation = data_dict["segmentation"]
        if isinstance(segmentation, (list, tuple)):
            raise RuntimeError(
                "TubedSkeletonTargetTransform must run before deep-supervision downsampling."
            )

        label_map = segmentation[0].detach().cpu().numpy()
        foreground = label_map > 0

        if foreground.any():
            skeleton = skeletonize(foreground)
            if self.radius > 0:
                skeleton = dilation(skeleton, footprint=diamond(self.radius))
            class_skeleton = skeleton.astype(np.int16) * label_map.astype(np.int16)
        else:
            class_skeleton = np.zeros_like(label_map, dtype=np.int16)

        data_dict["skel"] = torch.from_numpy(class_skeleton[None]).to(
            dtype=segmentation.dtype
        )
        return data_dict


class DownsampleSkeletonForDSTransform(BasicTransform):
    """Downsample the tubed skeleton to the same deep-supervision scales as GT."""

    def __init__(self, ds_scales: Union[List, Tuple]):
        super().__init__()
        self.ds_scales = ds_scales

    def apply(self, data_dict, **params):
        skeleton = data_dict["skel"]
        dtype = skeleton.dtype
        skeleton_float = None
        results = []

        for scale in self.ds_scales:
            if not isinstance(scale, (tuple, list)):
                scale = [scale] * (skeleton.ndim - 1)
            if all(value == 1 for value in scale):
                results.append(skeleton)
                continue

            new_shape = [
                round(size * factor)
                for size, factor in zip(skeleton.shape[1:], scale)
            ]
            if skeleton_float is None:
                skeleton_float = skeleton[None].float()
            downsampled = F.interpolate(
                skeleton_float,
                size=new_shape,
                mode="nearest-exact",
            )[0].to(dtype)
            results.append(downsampled)

        data_dict["skel"] = results
        return data_dict


class SkeletonContextFluxTargetTransform(BasicTransform):
    """Create the full-resolution DeepFlux-style target around class 1."""

    def __init__(self, radius: int = 7):
        super().__init__()
        if radius <= 0:
            raise ValueError("Flux context radius must be positive.")
        self.radius = int(radius)

    def apply(self, data_dict, **params):
        segmentation = data_dict["segmentation"]
        if isinstance(segmentation, (list, tuple)):
            raise RuntimeError(
                "SkeletonContextFluxTargetTransform must run before deep-supervision downsampling."
            )
        label_map = segmentation[0].detach().cpu().numpy()
        skeleton = label_map == 1
        flux = np.zeros((skeleton.ndim, *skeleton.shape), dtype=np.float32)
        context = np.zeros(skeleton.shape, dtype=bool)
        if skeleton.any():
            distance, nearest = ndi.distance_transform_edt(
                ~skeleton,
                return_distances=True,
                return_indices=True,
            )
            context = (distance > 0) & (distance <= self.radius)
            coordinates = np.indices(skeleton.shape)
            denominator = np.maximum(distance, 1e-6)
            for dimension in range(skeleton.ndim):
                component = (nearest[dimension] - coordinates[dimension]) / denominator
                flux[dimension, context] = component[context]

        data_dict["flux"] = torch.from_numpy(flux)
        data_dict["flux_mask"] = torch.from_numpy(context[None].astype(np.float32))
        return data_dict


class SemanticFluxAuxiliaryNetwork(nn.Module):
    """Wrap nnU-Net with a training-only full-resolution two-channel flux head."""

    def __init__(
        self,
        segmenter: nn.Module,
        num_segmentation_heads: int,
        spatial_dimensions: int,
    ):
        super().__init__()
        self.segmenter = segmenter
        if spatial_dimensions == 2:
            convolution = nn.Conv2d
        elif spatial_dimensions == 3:
            convolution = nn.Conv3d
        else:
            raise ValueError(
                f"Flux auxiliary head supports 2-D or 3-D data, got {spatial_dimensions}."
            )
        hidden_channels = max(8, num_segmentation_heads * 4)
        self.flux_head = nn.Sequential(
            convolution(num_segmentation_heads, hidden_channels, 3, padding=1),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            convolution(hidden_channels, spatial_dimensions, 1),
        )

    def forward(self, data):
        semantic = self.segmenter(data)
        # nnU-Net inference builds this same trainer class but switches the
        # network to eval mode. Returning only semantic outputs keeps the normal
        # predictor/exporter interface unchanged.
        if not self.training:
            return semantic
        full_resolution = semantic[0] if isinstance(semantic, (list, tuple)) else semantic
        return semantic, self.flux_head(full_resolution)


class nnUNetDataLoaderSkeletonRecall(nnUNetDataLoader):
    """nnU-Net 2.8.1 loader that keeps the extra skeleton target."""

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        data_all = None
        seg_all = None
        skel_all = None
        flux_all = None
        flux_mask_all = None

        with torch.no_grad():
            with threadpool_limits(limits=1, user_api=None):
                for batch_index, case_id in enumerate(selected_keys):
                    force_fg = self.get_do_oversample(batch_index)
                    data, seg, seg_prev, properties = self._data.load_case(case_id)
                    shape = data.shape[1:]

                    bbox_lbs, bbox_ubs = self.get_bbox(
                        shape,
                        force_fg,
                        properties["class_locations"],
                    )
                    bbox = [[lower, upper] for lower, upper in zip(bbox_lbs, bbox_ubs)]

                    data_cropped = torch.from_numpy(
                        crop_and_pad_nd(data, bbox, 0)
                    ).float()
                    seg_cropped = torch.from_numpy(
                        crop_and_pad_nd(
                            seg,
                            bbox,
                            -1,
                            cast_cropped_to=np.int16,
                        )
                    ).to(torch.int16)

                    if seg_prev is not None:
                        seg_prev_cropped = torch.from_numpy(
                            crop_and_pad_nd(
                                seg_prev,
                                bbox,
                                -1,
                                cast_cropped_to=np.int16,
                            )
                        ).to(torch.int16)
                        seg_cropped = torch.cat(
                            (seg_cropped, seg_prev_cropped[None]), dim=0
                        )

                    if self.patch_size_was_2d:
                        data_cropped = data_cropped[:, 0]
                        seg_cropped = seg_cropped[:, 0]

                    transformed = self.transforms(
                        **{
                            "image": data_cropped,
                            "segmentation": seg_cropped,
                        }
                    )
                    data_sample = transformed["image"]
                    seg_sample = transformed["segmentation"]
                    skel_sample = transformed["skel"]
                    flux_sample = transformed.get("flux")
                    flux_mask_sample = transformed.get("flux_mask")

                    if data_all is None:
                        data_all = torch.empty(
                            (self.batch_size, *data_sample.shape),
                            dtype=torch.float32,
                        )
                    data_all[batch_index] = data_sample

                    if flux_sample is not None:
                        if flux_all is None:
                            flux_all = torch.empty(
                                (self.batch_size, *flux_sample.shape),
                                dtype=flux_sample.dtype,
                            )
                            flux_mask_all = torch.empty(
                                (self.batch_size, *flux_mask_sample.shape),
                                dtype=flux_mask_sample.dtype,
                            )
                        flux_all[batch_index] = flux_sample
                        flux_mask_all[batch_index] = flux_mask_sample

                    if isinstance(seg_sample, list):
                        if seg_all is None:
                            seg_all = [
                                torch.empty(
                                    (self.batch_size, *sample.shape),
                                    dtype=sample.dtype,
                                )
                                for sample in seg_sample
                            ]
                            skel_all = [
                                torch.empty(
                                    (self.batch_size, *sample.shape),
                                    dtype=sample.dtype,
                                )
                                for sample in skel_sample
                            ]
                        for scale_index, sample in enumerate(seg_sample):
                            seg_all[scale_index][batch_index] = sample
                            skel_all[scale_index][batch_index] = skel_sample[scale_index]
                    else:
                        if seg_all is None:
                            seg_all = torch.empty(
                                (self.batch_size, *seg_sample.shape),
                                dtype=seg_sample.dtype,
                            )
                            skel_all = torch.empty(
                                (self.batch_size, *skel_sample.shape),
                                dtype=skel_sample.dtype,
                            )
                        seg_all[batch_index] = seg_sample
                        skel_all[batch_index] = skel_sample

        result = {
            "data": data_all,
            "target": seg_all,
            "skel": skel_all,
            "keys": selected_keys,
        }
        if flux_all is not None:
            result["flux"] = flux_all
            result["flux_mask"] = flux_mask_all
        return result


class SoftSkeletonRecallLoss(nn.Module):
    """Soft recall of foreground probabilities on the class-labelled GT skeleton."""

    def __init__(
        self,
        apply_nonlin=None,
        batch_dice: bool = False,
        do_bg: bool = False,
        smooth: float = 1e-5,
        ddp: bool = False,
        class_weights: Tuple[float, ...] | None = None,
    ):
        super().__init__()
        if do_bg:
            raise ValueError("Skeleton Recall Loss excludes the background class.")
        self.apply_nonlin = apply_nonlin
        self.batch_dice = batch_dice
        self.smooth = smooth
        self.ddp = ddp

        # Weights refer only to foreground classes in their dataset.json
        # order: class 1 (Skeleton/TM), then class 2 (Soma).
        if class_weights is None:
            self.class_weights = None
        else:
            weights = torch.as_tensor(class_weights, dtype=torch.float32)
            if weights.ndim != 1 or weights.numel() == 0:
                raise ValueError("class_weights must be a non-empty 1D sequence.")
            if torch.any(weights <= 0):
                raise ValueError("class_weights must contain only positive values.")
            self.register_buffer("class_weights", weights, persistent=False)

    def forward(self, logits, class_skeleton, loss_mask=None):
        probabilities = (
            self.apply_nonlin(logits) if self.apply_nonlin is not None else logits
        )
        probabilities = probabilities[:, 1:]
        axes = tuple(range(2, logits.ndim))

        with torch.no_grad():
            if class_skeleton.ndim != logits.ndim:
                class_skeleton = class_skeleton.view(
                    class_skeleton.shape[0],
                    1,
                    *class_skeleton.shape[1:],
                )
            one_hot = torch.zeros(
                logits.shape,
                device=logits.device,
                dtype=class_skeleton.dtype,
            )
            one_hot.scatter_(1, class_skeleton.long(), 1)
            one_hot = one_hot[:, 1:]
            denominator = (
                one_hot.sum(axes)
                if loss_mask is None
                else (one_hot * loss_mask).sum(axes)
            )

        numerator = (
            (probabilities * one_hot).sum(axes)
            if loss_mask is None
            else (probabilities * one_hot * loss_mask).sum(axes)
        )

        if self.batch_dice:
            numerator = numerator.sum(0)
            denominator = denominator.sum(0)

        recall = (numerator + self.smooth) / torch.clamp(
            denominator + self.smooth,
            min=1e-8,
        )

        if self.class_weights is None:
            return -recall.mean()

        class_weights = self.class_weights.to(
            device=recall.device,
            dtype=recall.dtype,
        )
        if recall.shape[-1] != class_weights.numel():
            raise RuntimeError(
                "Skeleton Recall class_weights do not match the number of "
                f"foreground classes: {class_weights.numel()} weights for "
                f"{recall.shape[-1]} classes."
            )

        # Keep the term on the same approximate scale as the unweighted mean.
        # For [2, 1], this is (2*SkeletonRecall + SomaRecall) / 3.
        weighted_recall = (
            recall * class_weights
        ).sum(dim=-1) / class_weights.sum()
        return -weighted_recall.mean()


class DiceSkeletonRecallCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        soft_dice_kwargs,
        soft_skeleton_kwargs,
        weight_skeleton_recall: float,
        ignore_label=None,
    ):
        super().__init__()
        if ignore_label is not None:
            raise NotImplementedError(
                "This trainer is intentionally restricted to fully annotated labels."
            )
        self.weight_skeleton_recall = float(weight_skeleton_recall)
        self.ce = RobustCrossEntropyLoss()
        self.dc = MemoryEfficientSoftDiceLoss(
            apply_nonlin=softmax_helper_dim1,
            **soft_dice_kwargs,
        )
        self.skeleton_recall = SoftSkeletonRecallLoss(
            apply_nonlin=softmax_helper_dim1,
            **soft_skeleton_kwargs,
        )

    def forward(self, logits, target, class_skeleton):
        dice_loss = self.dc(logits, target)
        ce_loss = self.ce(logits, target[:, 0])
        skeleton_loss = self.skeleton_recall(logits, class_skeleton)
        return dice_loss + ce_loss + self.weight_skeleton_recall * skeleton_loss


class nnUNetTrainerSkeletonRecallCells(nnUNetTrainer):
    """Paper-faithful Skeleton Recall trainer for labels 0/1/2."""

    skeleton_recall_weight = 1.0
    # None preserves the original equal mean across foreground classes.
    skeleton_recall_class_weights = None
    skeleton_tube_radius = 2
    full_resolution_loss_only = False

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        if self.label_manager.has_regions:
            raise NotImplementedError("Region-based training is not supported here.")

    def _build_loss(self):
        common_kwargs = {
            "batch_dice": self.configuration_manager.batch_dice,
            "smooth": 1e-5,
            "do_bg": False,
            "ddp": self.is_ddp,
        }
        loss = DiceSkeletonRecallCrossEntropyLoss(
            soft_dice_kwargs=common_kwargs,
            soft_skeleton_kwargs={
                **common_kwargs,
                "class_weights": self.skeleton_recall_class_weights,
            },
            weight_skeleton_recall=self.skeleton_recall_weight,
            ignore_label=self.label_manager.ignore_label,
        )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = deep_supervision_loss_weights(
                len(deep_supervision_scales),
                full_resolution_only=self.full_resolution_loss_only,
                tiny_last_weight=self.is_ddp and not self._do_i_compile(),
            )
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    @staticmethod
    def _add_skeleton_transforms(base_transforms, deep_supervision_scales):
        skeleton_transform = TubedSkeletonTargetTransform(
            radius=nnUNetTrainerSkeletonRecallCells.skeleton_tube_radius
        )
        downsample_index = next(
            (
                index
                for index, transform in enumerate(base_transforms.transforms)
                if isinstance(transform, DownsampleSegForDSTransform)
            ),
            None,
        )

        if downsample_index is None:
            base_transforms.transforms.append(skeleton_transform)
        else:
            base_transforms.transforms.insert(downsample_index, skeleton_transform)
            base_transforms.transforms.insert(
                downsample_index + 2,
                DownsampleSkeletonForDSTransform(deep_supervision_scales),
            )
        return base_transforms

    @staticmethod
    def get_training_transforms(
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: RandomScalar,
        deep_supervision_scales: Union[List, Tuple, None],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        use_mask_for_norm: List[bool] = None,
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> BasicTransform:
        transforms = nnUNetTrainer.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm,
            is_cascaded,
            foreground_labels,
            regions,
            ignore_label,
        )
        return nnUNetTrainerSkeletonRecallCells._add_skeleton_transforms(
            transforms,
            deep_supervision_scales,
        )

    @staticmethod
    def get_validation_transforms(
        deep_supervision_scales: Union[List, Tuple, None],
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> BasicTransform:
        transforms = nnUNetTrainer.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded,
            foreground_labels,
            regions,
            ignore_label,
        )
        return nnUNetTrainerSkeletonRecallCells._add_skeleton_transforms(
            transforms,
            deep_supervision_scales,
        )

    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=(
                self.label_manager.foreground_regions
                if self.label_manager.has_regions
                else None
            ),
            ignore_label=self.label_manager.ignore_label,
        )
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=(
                self.label_manager.foreground_regions
                if self.label_manager.has_regions
                else None
            ),
            ignore_label=self.label_manager.ignore_label,
        )

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        loader_kwargs = {
            "label_manager": self.label_manager,
            "oversample_foreground_percent": self.oversample_foreground_percent,
            "sampling_probabilities": None,
            "pad_sides": None,
            "probabilistic_oversampling": self.probabilistic_oversampling,
        }
        dl_tr = nnUNetDataLoaderSkeletonRecall(
            dataset_tr,
            self.batch_size,
            initial_patch_size,
            self.configuration_manager.patch_size,
            transforms=tr_transforms,
            **loader_kwargs,
        )
        dl_val = nnUNetDataLoaderSkeletonRecall(
            dataset_val,
            self.batch_size,
            self.configuration_manager.patch_size,
            self.configuration_manager.patch_size,
            transforms=val_transforms,
            **loader_kwargs,
        )

        allowed_processes = get_allowed_n_proc_DA()
        if allowed_processes == 0:
            train_generator = SingleThreadedAugmenter(dl_tr, None)
            val_generator = SingleThreadedAugmenter(dl_val, None)
        else:
            train_generator = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr,
                transform=None,
                num_processes=allowed_processes,
                num_cached=max(6, allowed_processes // 2),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
            val_generator = NonDetMultiThreadedAugmenter(
                data_loader=dl_val,
                transform=None,
                num_processes=max(1, allowed_processes // 2),
                num_cached=max(3, allowed_processes // 4),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )

        _ = next(train_generator)
        _ = next(val_generator)
        return train_generator, val_generator

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        skeleton = batch["skel"]

        if isinstance(target, list):
            target = [item.to(self.device, non_blocking=True) for item in target]
            skeleton = [item.to(self.device, non_blocking=True) for item in skeleton]
        else:
            target = target.to(self.device, non_blocking=True)
            skeleton = skeleton.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(data)
            loss = self.loss(output, target, skeleton)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": loss.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        skeleton = batch["skel"]

        if isinstance(target, list):
            target = [item.to(self.device, non_blocking=True) for item in target]
            skeleton = [item.to(self.device, non_blocking=True) for item in skeleton]
        else:
            target = target.to(self.device, non_blocking=True)
            skeleton = skeleton.to(self.device, non_blocking=True)

        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(data)
            del data
            loss = self.loss(output, target, skeleton)

        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        axes = [0] + list(range(2, output.ndim))
        output_seg = output.argmax(1)[:, None]
        predicted_one_hot = torch.zeros(
            output.shape,
            device=output.device,
            dtype=torch.float16,
        )
        predicted_one_hot.scatter_(1, output_seg, 1)
        del output_seg

        tp, fp, fn, _ = get_tp_fp_fn_tn(
            predicted_one_hot,
            target,
            axes=axes,
            mask=None,
        )
        tp_hard = tp.detach().cpu().numpy()[1:]
        fp_hard = fp.detach().cpu().numpy()[1:]
        fn_hard = fn.detach().cpu().numpy()[1:]

        return {
            "loss": loss.detach().cpu().numpy(),
            "tp_hard": tp_hard,
            "fp_hard": fp_hard,
            "fn_hard": fn_hard,
        }


# SKELETON_TM_CLASS_WEIGHTING_V1

class nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1x(
    nnUNetTrainerSkeletonRecallCells
):
    """Prioritize class 1 Skeleton/TM recall 2:1 over class 2 Soma recall."""

    # Background is excluded. Dataset 136 foreground order is:
    # [Skeleton/TM (class 1), Soma (class 2)].
    skeleton_recall_class_weights = (2.0, 1.0)


class nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xFullResOnly(
    nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1x
):
    """Experiment R1: apply the complete loss only at full resolution."""

    full_resolution_loss_only = True


class nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xFullResOnlyDebug50(
    nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xFullResOnly
):
    """Short R1 smoke test with a separate result folder."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        # nnUNetTrainer introspects the concrete __init__ signature and expects
        # these names to exist in its own constructor locals. Do not replace
        # this explicit signature with *args/**kwargs.
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 50


def add_label_safe_spatial_augmentation(
    transforms: BasicTransform,
    patch_size: Union[np.ndarray, Tuple[int]],
) -> BasicTransform:
    """Replace interpolating rotation/scaling with an exact 2-D Rot90."""

    if len(patch_size) != 2:
        raise NotImplementedError(
            "The label-safe connectivity experiment is intentionally limited to 2-D."
        )
    spatial_index = next(
        (
            index
            for index, transform in enumerate(transforms.transforms)
            if isinstance(transform, SpatialTransform)
        ),
        None,
    )
    if spatial_index is None:
        raise RuntimeError("nnU-Net training transforms contain no SpatialTransform.")
    spatial = transforms.transforms[spatial_index]
    spatial.p_rotation = 0.0
    spatial.p_scaling = 0.0
    transforms.transforms.insert(
        spatial_index + 1,
        RandomTransform(
            Rot90Transform(
                num_axis_combinations=1,
                num_rot_per_combination=(1, 2, 3),
                allowed_axes={0, 1},
            ),
            apply_probability=0.5,
        ),
    )
    return transforms


def add_flux_target_transform(
    transforms: BasicTransform,
    radius: int,
) -> BasicTransform:
    """Insert the full-resolution flux target before skeleton/DS transforms."""

    insert_index = next(
        (
            index
            for index, transform in enumerate(transforms.transforms)
            if isinstance(transform, TubedSkeletonTargetTransform)
        ),
        None,
    )
    if insert_index is None:
        insert_index = next(
            (
                index
                for index, transform in enumerate(transforms.transforms)
                if isinstance(transform, DownsampleSegForDSTransform)
            ),
            len(transforms.transforms),
        )
    transforms.transforms.insert(
        insert_index,
        SkeletonContextFluxTargetTransform(radius=radius),
    )
    return transforms


class nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAug(
    nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1x
):
    """Experiment R4: exact Rot90/mirroring without interpolating label warps."""

    @staticmethod
    def get_training_transforms(
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: RandomScalar,
        deep_supervision_scales: Union[List, Tuple, None],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        use_mask_for_norm: List[bool] = None,
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> BasicTransform:
        transforms = nnUNetTrainerSkeletonRecallCells.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm,
            is_cascaded,
            foreground_labels,
            regions,
            ignore_label,
        )
        return add_label_safe_spatial_augmentation(transforms, patch_size)


class nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAugDebug50(
    nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAug
):
    """Short R4 smoke test with an isolated result folder."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 50


class nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAugMATSFineTune100(
    nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAug
):
    """Fine-tune Dataset139 weights on full MATS overview images."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # One tenth of the original learning rate protects the already learned
        # single-cell representation while adapting it to overview imagery.
        self.initial_lr = 1e-3
        self.num_epochs = 100

    def initialize(self):
        if self.was_initialized:
            return
        super().initialize()
        checkpoint_path = os.environ.get("NNUNET_MATS_FINETUNE_CHECKPOINT", "")
        if not checkpoint_path:
            return

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        source_weights = checkpoint["network_weights"]
        target_network = self.network
        if hasattr(target_network, "module"):
            target_network = target_network.module
        if hasattr(target_network, "_orig_mod"):
            target_network = target_network._orig_mod
        target_keys = set(target_network.state_dict())
        transferred = {}
        for key, value in source_weights.items():
            normalized_key = key[7:] if key.startswith("module.") and key[7:] in target_keys else key
            transferred[normalized_key] = value
        target_network.load_state_dict(transferred, strict=True)
        self.print_to_log_file(
            "Loaded the complete Dataset139 network for MATS fine-tuning, "
            "including all segmentation heads:",
            checkpoint_path,
        )

    def on_epoch_end(self):
        """Do not abort training if only the optional progress PNG fails."""

        plot_progress_png = self.logger.plot_progress_png

        def safe_plot_progress_png(output_folder):
            try:
                plot_progress_png(output_folder)
            except OSError as error:
                self.print_to_log_file(
                    "WARNING: progress.png could not be updated; training and "
                    "checkpointing continue:",
                    repr(error),
                )

        self.logger.plot_progress_png = safe_plot_progress_png
        try:
            super().on_epoch_end()
        finally:
            self.logger.plot_progress_png = plot_progress_png


class nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAugMATSFineTune5(
    nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAugMATSFineTune100
):
    """Five-epoch technical check in a separate result directory."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 5


class nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAugSyntheticMultiCellFineTune200(
    nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAugMATSFineTune100
):
    """Fine-tune the complete Dataset141 model on Dataset143."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-3
        self.num_epochs = 200

    def initialize(self):
        if self.was_initialized:
            return
        # Deliberately bypass the MATS checkpoint loader inherited only for its
        # robust progress plotting. Dataset143 must start from Dataset141.
        nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAug.initialize(self)
        checkpoint_path = os.environ.get(
            "NNUNET_SYNTHETIC_MULTICELL_FINETUNE_CHECKPOINT", ""
        )
        if not checkpoint_path:
            return

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        source_weights = checkpoint["network_weights"]
        target_network = self.network
        if hasattr(target_network, "module"):
            target_network = target_network.module
        if hasattr(target_network, "_orig_mod"):
            target_network = target_network._orig_mod
        target_keys = set(target_network.state_dict())
        transferred = {}
        for key, value in source_weights.items():
            normalized_key = (
                key[7:]
                if key.startswith("module.") and key[7:] in target_keys
                else key
            )
            transferred[normalized_key] = value
        target_network.load_state_dict(transferred, strict=True)
        self.print_to_log_file(
            "Loaded the complete Dataset141 network for synthetic multi-cell "
            "fine-tuning, including all segmentation heads:",
            checkpoint_path,
        )


class nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAugSyntheticMultiCellFineTune20(
    nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAugSyntheticMultiCellFineTune200
):
    """Twenty-epoch Dataset143 technical check in a separate result folder."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 20


class DeepFluxAuxTrainingMixin:
    """Training-only DeepFlux-style auxiliary regression for class 1."""

    flux_context_radius = 7
    flux_auxiliary_weight = 0.1

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        segmenter = nnUNetTrainer.build_network_architecture(
            plans_manager,
            configuration_manager,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        )
        return SemanticFluxAuxiliaryNetwork(
            segmenter,
            num_output_channels,
            spatial_dimensions=len(configuration_manager.patch_size),
        )

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        skeleton = batch["skel"]
        flux_target = batch["flux"].to(self.device, non_blocking=True)
        flux_mask = batch["flux_mask"].to(self.device, non_blocking=True)

        if isinstance(target, list):
            target = [item.to(self.device, non_blocking=True) for item in target]
            skeleton = [item.to(self.device, non_blocking=True) for item in skeleton]
        else:
            target = target.to(self.device, non_blocking=True)
            skeleton = skeleton.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            semantic_output, predicted_flux = self.network(data)
            semantic_loss = self.loss(semantic_output, target, skeleton)
            flux_error = F.smooth_l1_loss(
                predicted_flux,
                flux_target,
                reduction="none",
            ).sum(dim=1, keepdim=True)
            flux_denominator = torch.clamp(flux_mask.sum(), min=1.0)
            flux_loss = (flux_error * flux_mask).sum() / flux_denominator
            loss = semantic_loss + self.flux_auxiliary_weight * flux_loss

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {
            "loss": loss.detach().cpu().numpy(),
            "semantic_loss": semantic_loss.detach().cpu().numpy(),
            "flux_loss": flux_loss.detach().cpu().numpy(),
        }


class nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAugFluxAux(
    DeepFluxAuxTrainingMixin,
    nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAug,
):
    """Experiment R5: R4 plus a full-resolution radius-7 flux auxiliary head."""

    @staticmethod
    def get_training_transforms(
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: RandomScalar,
        deep_supervision_scales: Union[List, Tuple, None],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        use_mask_for_norm: List[bool] = None,
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> BasicTransform:
        transforms = (
            nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAug.get_training_transforms(
                patch_size,
                rotation_for_DA,
                deep_supervision_scales,
                mirror_axes,
                do_dummy_2d_data_aug,
                use_mask_for_norm,
                is_cascaded,
                foreground_labels,
                regions,
                ignore_label,
            )
        )
        return add_flux_target_transform(
            transforms,
            radius=DeepFluxAuxTrainingMixin.flux_context_radius,
        )


class nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAugFluxAuxDebug50(
    nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAugFluxAux
):
    """Short R5 smoke test with an isolated result folder."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 50


class nnUNetTrainerSkeletonRecallCellsW01(nnUNetTrainerSkeletonRecallCells):
    """Conservative paper setting with Skeleton Recall weight 0.1."""

    skeleton_recall_weight = 0.1
