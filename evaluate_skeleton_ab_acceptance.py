from __future__ import annotations

"""Evaluate a completed blinded skeleton A/B CSV against the staged gates."""

import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "continuity",
    "false_connections",
    "soma_attachment",
    "one_pixel_width",
    "overall",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--base-name", required=True)
    parser.add_argument(
        "--automatic-summary",
        type=Path,
        help="Optional candidate summary.json from skeleton_connectivity_experiments.py.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def read_ratings(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        sample = stream.read(4096)
        stream.seek(0)
        delimiter = ";" if sample.count(";") >= sample.count(",") else ","
        rows = list(csv.DictReader(stream, delimiter=delimiter))
    if not rows:
        raise ValueError(f"Ratings CSV is empty: {path}")
    incomplete = [row.get("case_id", "?") for row in rows if row.get("complete") != "yes"]
    if incomplete:
        raise ValueError(
            f"Ratings contain {len(incomplete)} incomplete cases: {', '.join(incomplete[:10])}"
        )
    return rows


def evaluate(
    rows: list[dict[str, str]],
    candidate_name: str,
    base_name: str,
    automatic_summary: Path | None,
) -> dict[str, object]:
    counts: dict[str, dict[str, int]] = {}
    for field in FIELDS:
        column = f"{field}_actual_winner"
        if column not in rows[0]:
            raise ValueError(f"Ratings CSV is missing column {column!r}")
        values = [row[column].strip() for row in rows]
        unexpected = sorted(set(values) - {candidate_name, base_name, "tie"})
        if unexpected:
            raise ValueError(
                f"Unexpected model names in {column}: {unexpected}. Expected "
                f"{candidate_name!r}, {base_name!r}, or 'tie'."
            )
        counts[field] = {
            "candidate": values.count(candidate_name),
            "base": values.count(base_name),
            "tie": values.count("tie"),
        }

    manual_gates = {
        "continuity_net_wins_at_least_8": (
            counts["continuity"]["candidate"] - counts["continuity"]["base"] >= 8
        ),
        "false_connections_not_worse": (
            counts["false_connections"]["candidate"]
            >= counts["false_connections"]["base"]
        ),
        "soma_attachment_not_worse": (
            counts["soma_attachment"]["candidate"]
            >= counts["soma_attachment"]["base"]
        ),
        "one_pixel_width_not_worse": (
            counts["one_pixel_width"]["candidate"]
            >= counts["one_pixel_width"]["base"]
        ),
        "overall_not_worse": (
            counts["overall"]["candidate"] >= counts["overall"]["base"]
        ),
    }

    automatic: dict[str, object] | None = None
    automatic_pass = True
    if automatic_summary is not None:
        payload = json.loads(automatic_summary.read_text(encoding="utf-8"))
        automatic = payload.get("summary", payload)
        automatic_pass = bool(automatic.get("automatic_gate_pass", False))

    return {
        "ratings_file": str(Path(rows[0].get("_source", ""))) if "_source" in rows[0] else None,
        "cases": len(rows),
        "candidate_name": candidate_name,
        "base_name": base_name,
        "counts": counts,
        "manual_gates": manual_gates,
        "manual_gate_pass": all(manual_gates.values()),
        "automatic_summary": automatic,
        "automatic_gate_pass": automatic_pass,
        "accepted": all(manual_gates.values()) and automatic_pass,
    }


def main() -> None:
    args = parse_args()
    rows = read_ratings(args.ratings)
    result = evaluate(rows, args.candidate_name, args.base_name, args.automatic_summary)
    result["ratings_file"] = str(args.ratings.resolve())
    output = args.output or args.ratings.with_name(
        f"{args.ratings.stem}-acceptance.json"
    )
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Cases: {result['cases']}")
    for field in FIELDS:
        count = result["counts"][field]
        print(
            f"{field:20} candidate={count['candidate']:>3}  "
            f"base={count['base']:>3}  tie={count['tie']:>3}"
        )
    print("Manual gates:")
    for name, passed in result["manual_gates"].items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print(f"Automatic gate: {'PASS' if result['automatic_gate_pass'] else 'FAIL'}")
    print(f"Decision: {'ACCEPT CANDIDATE' if result['accepted'] else 'KEEP CURRENT BEST'}")
    print(f"Written: {output.resolve()}")


if __name__ == "__main__":
    main()
