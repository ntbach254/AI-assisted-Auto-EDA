import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

print("STARTING num_cat.py")
class NumCatAssociations:
    def __init__(self, min_n=30, max_categories=30):
        self.min_n = int(min_n)
        self.max_categories = int(max_categories)

    def _eta_squared(self, groups):
        all_vals = np.concatenate(groups)
        grand_mean = np.mean(all_vals)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
        ss_total = sum((v - grand_mean) ** 2 for v in all_vals)
        if ss_total <= 0:
            return 0.0
        return float(ss_between / ss_total)

    def _rank_biserial(self, a, b):
        na, nb = len(a), len(b)
        if na == 0 or nb == 0:
            return 0.0, 1.0
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        rbc = 1.0 - (2.0 * u) / (na * nb)
        return float(rbc), float(p)

    def compute(self, df, profile, num_col, cat_col):
        pair = pd.concat([df[num_col], df[cat_col]], axis=1).dropna()
        n = int(len(pair))
        if n < self.min_n:
            return None

        x = pair.iloc[:, 0].astype(float)
        g = pair.iloc[:, 1].astype(str)

        if x.nunique() < 2 or g.nunique() < 2:
            return None

        k = int(g.nunique())
        if k > self.max_categories:
            return {
                "var1": num_col,
                "var2": cat_col,
                "type": "num_cat",
                "relationship": "group_difference",
                "test": "skipped",
                "effect": np.nan,
                "effect_name": "eta2",
                "p_value": np.nan,
                "n": n,
                "notes": f"too_many_categories_{k}",
            }

        levels = sorted(g.unique())
        groups = [x[g == lvl].to_numpy() for lvl in levels]
        sizes = [len(arr) for arr in groups]
        if min(sizes) < 2:
            return None

        if k == 2:
            a = groups[0]
            b = groups[1]
            effect, p = self._rank_biserial(a, b)
            return {
                "var1": num_col,
                "var2": cat_col,
                "type": "num_cat",
                "relationship": "group_difference",
                "test": "mannwhitneyu",
                "effect": float(effect),
                "effect_name": "rank_biserial",
                "p_value": float(p),
                "n": n,
                "notes": "nonparametric_preferred",
            }

        h, p = stats.kruskal(*groups)
        eta2 = self._eta_squared(groups)
        return {
            "var1": num_col,
            "var2": cat_col,
            "type": "num_cat",
            "relationship": "group_difference",
            "test": "kruskal",
            "effect": float(eta2),
            "effect_name": "eta2_approx",
            "p_value": float(p),
            "n": n,
            "notes": "nonparametric_preferred",
        }


def load_profile(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_numeric_feature(df, profile, col):
    ft = profile.get(col, {}).get("feature_type")
    return ft in {"numeric_continuous", "numeric_ordinal"} and pd.api.types.is_numeric_dtype(df[col])


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

    numeric_cols = [c for c in df.columns if is_numeric_feature(df, profile, c)]
    categorical_cols = [c for c in df.columns if is_categorical_feature(profile, c)]

    engine = NumCatAssociations(
        min_n=args.min_n,
        max_categories=args.max_categories,
    )

    rows = []
    for num_col, cat_col in product(numeric_cols, categorical_cols):
        res = engine.compute(df, profile, num_col, cat_col)
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
    out_path = output_dir / f"{base_name}_num_cat_raw.csv"
    out_df.to_csv(out_path, index=False)

    print(f"Num-cat complete: {len(out_df)} rows")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()