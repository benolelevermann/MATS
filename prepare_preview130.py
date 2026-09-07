import json
import shutil
from pathlib import Path


PROJECT = Path(r"C:\Ole\20260721_CellClassification_v2")

RAW_ROOT = PROJECT / "nnUNet_raw"
PREPROCESSED_ROOT = PROJECT / "nnUNet_preprocessed"

PREVIEW_ROOT = PROJECT / "preview130"
PREVIEW_INPUT = PREVIEW_ROOT / "input"
PREVIEW_LABELS = PREVIEW_ROOT / "manual_labels"


def find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))

    if len(matches) != 1:
        raise RuntimeError(
            f"Erwartete genau einen Treffer für {pattern} in {root}, "
            f"gefunden: {matches}"
        )

    return matches[0]


def main():
    raw_dataset = find_one(
        RAW_ROOT,
        "Dataset130_*",
    )

    preprocessed_dataset = find_one(
        PREPROCESSED_ROOT,
        "Dataset130_*",
    )

    splits_path = (
        preprocessed_dataset
        / "splits_final.json"
    )

    with open(
        splits_path,
        "r",
        encoding="utf-8",
    ) as file:
        splits = json.load(file)

    # Sechs Validierungsfälle aus Fold 0
    case_ids = splits[0]["val"][:6]

    if PREVIEW_ROOT.exists():
        shutil.rmtree(PREVIEW_ROOT)

    PREVIEW_INPUT.mkdir(parents=True)
    PREVIEW_LABELS.mkdir(parents=True)

    for case_id in case_ids:
        for channel in (0, 1):
            source = (
                raw_dataset
                / "imagesTr"
                / f"{case_id}_{channel:04d}.tif"
            )

            target = (
                PREVIEW_INPUT
                / source.name
            )

            if not source.exists():
                raise FileNotFoundError(source)

            shutil.copy2(source, target)

        label_source = (
            raw_dataset
            / "labelsTr"
            / f"{case_id}.tif"
        )

        if not label_source.exists():
            raise FileNotFoundError(label_source)

        shutil.copy2(
            label_source,
            PREVIEW_LABELS / label_source.name,
        )

    print("Preview-Fälle:")
    for case_id in case_ids:
        print(case_id)

    print()
    print(f"Input:  {PREVIEW_INPUT}")
    print(f"Labels: {PREVIEW_LABELS}")


if __name__ == "__main__":
    main()
