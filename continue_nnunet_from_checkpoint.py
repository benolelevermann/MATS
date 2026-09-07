from __future__ import annotations

"""Continue one nnU-Net run from an explicitly selected checkpoint."""

import argparse
import importlib

import torch

from nnunetv2.run.run_training import get_trainer_from_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("configuration")
    parser.add_argument("fold", type=int)
    parser.add_argument("--trainer", required=True)
    parser.add_argument("--plans", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bootstrap-module")
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    parser.add_argument("--npz", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_module:
        importlib.import_module(args.bootstrap_module)
    device = torch.device(args.device)
    trainer = get_trainer_from_args(
        args.dataset,
        args.configuration,
        args.fold,
        args.trainer,
        args.plans,
        continue_training=True,
        device=device,
    )
    trainer.load_checkpoint(args.checkpoint)
    print(
        f"Continuing explicitly from {args.checkpoint} "
        f"at epoch {trainer.current_epoch} of {trainer.num_epochs}"
    )
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    trainer.run_training()
    trainer.perform_actual_validation(args.npz)


if __name__ == "__main__":
    main()
