"""CLI: generate the synthetic post-offer decline dataset.

Usage:
    uv run python scripts/generate_postoffer_data.py \
        --n-samples 3000 --target-rate 0.18 --seed 42

Runs in ~1 second. Writes data/processed/postoffer_v1.parquet.

Direct response to mid-sem viva feedback (Ask #1): predicting post-offer
drop-off has higher business value than mid-application. This dataset
supports the end-sem extension proof-of-concept.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from recruit.config import settings
from recruit.postoffer.simulation import (
    generate_post_offer_dataset,
    save_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=3000)
    parser.add_argument("--target-rate", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=settings.data_path / "processed" / "postoffer_v1.parquet",
    )
    args = parser.parse_args()

    print(f"Generating {args.n_samples:,} synthetic post-offer rows "
          f"(target decline rate {args.target_rate:.2f}, seed {args.seed})...")
    df = generate_post_offer_dataset(
        n_samples=args.n_samples,
        target_rate=args.target_rate,
        seed=args.seed,
    )
    save_dataset(df, args.out)


if __name__ == "__main__":
    main()
