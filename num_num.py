import argparse
import json
from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy import stats

print("STARTING num_num.py")
class NumNumAssociations:
    def __init__(
        self,
        min_n=30,
        prefer_spearman_if_skew_gt=1.0,
        prefer_spearman_if_outlier_rate_gt=0.10,
    ):
        self.min_n = int(min_n)
        self.prefer_spearman_if_skew_gt = float(prefer_spearman_if_skew_gt)
        self.prefer_spearman_if_outlier_rate_gt = float(prefer_spearman_if_outlier_rate_gt)

    def _choose_method(self, a, b, profile):
        pa = profile.get(a, {})
        pb = profile.get(b, {})

        skew_a = pa.get("skewness")
        skew_b = pb.get("skewness")
        out_a = pa.get("outlier_rate")
        out_b = pb.get("outlier_rate")

        skew_flag = False
        out_flag = False

        if skew_a is not None and abs(float(skew_a)) > self.prefer_spearman_if_skew_gt:
            skew_flag = True
        if skew_b is not None and abs(float(skew_b)) > self.prefer_spearman_if_skew_gt:
            skew_flag = True

        if out_a is not None and float(out_a) > self.prefer_spearman_if_outlier_rate_gt:
            out_flag = True
        if out_b is not None and float(out_b) > self.prefer_spearman_if_outlier_rate_gt:
            out_flag = True

        return "spearman" if (skew_flag or out_flag) else "pearson"

    def compute(self, df, profile, col_a, col_b):
        pair = pd.concat([df[col_a], df[col_b]], axis=1).dropna()
        n = int(len(pair))
        if n < self.min_n:
            return None

        x = pair.iloc[:, 0].astype(float)
        y = pair.iloc[:, 1].astype(float)

        if x.nunique() < 2 or y.nunique() < 2:
            return None

        method = self._choose_method(col_a, col_b, profile)

        if method == "spearman":
            rho, p = stats.spearmanr(x, y)
            return {
                "var1": col_a,
                "var2": col_b,
                "type": "num_num",
                "relationship": "monotonic",
                "test": "spearmanr",
                "effect": float(rho),
                "effect_name": "rho",
                "p_value": float(p),
                "n": n,
                "notes": "",
            }

        r, p = stats.pearsonr(x, y)
        return {
            "var1": col_a,
            "var2": col_b,
            "type": "num_num",
            "relationship": "linear",
            "test": "pearsonr",
            "effect": float(r),
            "effect_name": "r",
            "p_value": float(p),
            "n": n,
            "notes": "",
        }


def load_profile(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_numeric_feature(df, profile, col):
    ft = profile.get(col, {}).get("feature_type")
    return ft in {"numeric_continuous", "numeric_ordinal"} and pd.api.types.is_numeric_dtype(df[col])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--profile-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-n", type=int, default=30)
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

    engine = NumNumAssociations(min_n=args.min_n)

    rows = []
    for a, b in combinations(numeric_cols, 2):
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
    out_path = output_dir / f"{base_name}_num_num_raw.csv"
    out_df.to_csv(out_path, index=False)

    print(f"Num-num complete: {len(out_df)} rows")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()