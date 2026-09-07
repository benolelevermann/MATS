from __future__ import annotations

"""Build Dataset141 from Dataset139 plus newly approved single-cell crops."""

import argparse
import json
from pathlib import Path

from build_dataset139_from_dataset138_and_reviewed import build_dataset


SCRIPT_VERSION = "dataset141-dataset139-plus-net139-reviewed-v1-2026-09-04"
README = (
    "Dataset141 = unveraendertes Dataset139 plus die danach mit Netz 139 erzeugten "
    "und manuell freigegebenen Einzelzell-Crops. Bereits in Dataset139 enthaltene "
    "Review-Zellen werden per Bild-und-Label-Hash uebersprungen. Dataset140 und "
    "vollstaendige Uebersichtsbilder sind nicht enthalten. Der Dataset139-Split "
    "bleibt unveraendert; neue Zellen liegen ausschliesslich im Training.\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--approved-dataset", type=Path, required=True)
    parser.add_argument("--output-dataset", type=Path, required=True)
    parser.add_argument("--base-splits", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_dataset(
        args.base_dataset.resolve(),
        args.approved_dataset.resolve(),
        args.output_dataset.resolve(),
        args.base_splits.resolve(),
        base_origin="dataset139",
        script_version=SCRIPT_VERSION,
        split_output_name="splits_final_preserve_dataset139.json",
        readme_text=README,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
