import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bh_fdr import benjamini_hochberg

print("STARTING fdr_correction.py-------------------------------------------------------------------------------------------")

EXPECTED_COLUMNS = [
    "var1", "var2", "type", "relationship", "test",
    "effect", "effect_name", "p_value", "n", "notes"
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-n", type=int, default=30)
    parser.add_argument("--min-abs-r", type=float, default=0.15)
    parser.add_argument("--min-cramers-v", type=float, default=0.20)
    parser.add_argument("--min-eta2", type=float, default=0.05)
    parser.add_argument("--min-rank-biserial", type=float, default=0.15)
    return parser.parse_args()


def safe_read_csv(path: Path):
    if not path.exists():
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    if df.empty:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    return df[EXPECTED_COLUMNS]


def keep_row(row, min_n, min_abs_r, min_cramers_v, min_eta2, min_rank_biserial):
    if not bool(row["significant"]):
        return False

    if int(row["n"]) < min_n:
        return False

    t = row["type"]
    eff = float(row["effect"])

    if t == "num_num":
        return abs(eff) >= min_abs_r

    if t == "cat_cat":
        return eff >= min_cramers_v

    if t == "num_cat":
        if row["effect_name"] == "rank_biserial":
            return abs(eff) >= min_rank_biserial
        return eff >= min_eta2

    return False


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = args.dataset_name

    num_num = safe_read_csv(input_dir / f"{dataset}_num_num_raw.csv")
    num_cat = safe_read_csv(input_dir / f"{dataset}_num_cat_raw.csv")
    cat_cat = safe_read_csv(input_dir / f"{dataset}_cat_cat_raw.csv")

    full = pd.concat([num_num, num_cat, cat_cat], ignore_index=True)

    if full.empty:
        full = pd.DataFrame(columns=EXPECTED_COLUMNS + ["adjusted_p", "significant"])
        filtered = full.copy()
    else:
        full["p_value"] = pd.to_numeric(full["p_value"], errors="coerce")
        full["effect"] = pd.to_numeric(full["effect"], errors="coerce")
        full["n"] = pd.to_numeric(full["n"], errors="coerce")

        valid = full["p_value"].notna() & np.isfinite(full["p_value"].to_numpy(dtype=float))
        pvals = full.loc[valid, "p_value"].to_numpy(dtype=float)

        full["adjusted_p"] = np.nan
        full["significant"] = False

        if len(pvals) > 0:
            qvals = benjamini_hochberg(pvals)
            full.loc[valid, "adjusted_p"] = qvals
            full.loc[valid, "significant"] = full.loc[valid, "adjusted_p"] < args.alpha

        mask = full.apply(
            keep_row,
            axis=1,
            min_n=args.min_n,
            min_abs_r=args.min_abs_r,
            min_cramers_v=args.min_cramers_v,
            min_eta2=args.min_eta2,
            min_rank_biserial=args.min_rank_biserial,
        )
        filtered = full[mask].reset_index(drop=True)

    full_path = output_dir / f"{dataset}_all_associations_fdr.csv"
    filtered_path = output_dir / f"{dataset}_associations_filtered.csv"

    full.to_csv(full_path, index=False)
    filtered.to_csv(filtered_path, index=False)

    print(f"FDR complete: {dataset}")
    print(f"All rows: {len(full)}")
    print(f"Filtered rows: {len(filtered)}")
    print(f"Saved: {full_path}")
    print(f"Saved: {filtered_path}")


if __name__ == "__main__":
    main()