import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

print("STARTING cat_cat.py")
class CatCatAssociations:
    def __init__(self, min_n=30, min_expected_count=5, max_categories=30):
        self.min_n = int(min_n)
        self.min_expected_count = float(min_expected_count)
        self.max_categories = int(max_categories)

    def compute(self, df, profile, col_a, col_b):
        pair = pd.concat([df[col_a], df[col_b]], axis=1).dropna()
        n = int(len(pair))
        if n < self.min_n:
            return None

        ka = int(pair.iloc[:, 0].nunique())
        kb = int(pair.iloc[:, 1].nunique())
        if ka > self.max_categories or kb > self.max_categories:
            return {
                "var1": col_a,
                "var2": col_b,
                "type": "cat_cat",
                "relationship": "association",
                "test": "skipped",
                "effect": np.nan,
                "effect_name": "cramers_v",
                "p_value": np.nan,
                "n": n,
                "notes": f"too_many_categories_{ka}x{kb}",
            }

        contingency = pd.crosstab(pair.iloc[:, 0].astype(str), pair.iloc[:, 1].astype(str))
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            return None

        chi2, p, _, expected = stats.chi2_contingency(contingency, correction=False)

        note = ""
        if (expected < self.min_expected_count).mean() > 0.20:
            note = "sparse_table_expected_counts_low"

        r, c = contingency.shape
        denom = n * min(r - 1, c - 1)
        v = 0.0 if denom <= 0 else float(np.sqrt(chi2 / denom))

        return {
            "var1": col_a,
            "var2": col_b,
            "type": "cat_cat",
            "relationship": "association",
            "test": "chi2",
            "effect": float(v),
            "effect_name": "cramers_v",
            "p_value": float(p),
            "n": n,
            "notes": note,
        }


def load_profile(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_categorical_feature(profile, col):
    ft = profile.get(col, {}).get("feature_type")
    return ft in {"categorical_low_cardinality", "categorical_high_cardinality", "binary"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--profile-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-n", type=int, default=30)
    parser.add_argument("--max-categories", type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()

    input_csv = Path(args.input_csv)
    profile_json = Path(args.profile_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = input_csv.stem.replace("_cleaned", "")

    df = pd.read_csv(input_csv)
    profile = load_profile(profile_json)

    categorical_cols = [c for c in df.columns if is_categorical_feature(profile, c)]

    engine = CatCatAssociations(
        min_n=args.min_n,
        max_categories=args.max_categories,
    )

    rows = []
    for a, b in combinations(categorical_cols, 2):
        res = engine.compute(df, profile, a, b)
        if res is not None:
            rows.append(res)

    EXPECTED_COLUMNS = [
        "var1",
        "var2",
        "type",
        "relationship",
        "test",
        "effect",
        "effect_name",
        "p_value",
        "n",
        "notes",
    ]
    out_df = pd.DataFrame(rows, columns = EXPECTED_COLUMNS)
    out_path = output_dir / f"{base_name}_cat_cat_raw.csv"
    out_df.to_csv(out_path, index=False)

    print(f"Cat-cat complete: {len(out_df)} rows")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()