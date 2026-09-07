import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw input CSV")
    parser.add_argument("--output-dir", required=True, help="Directory to save train/test CSVs")
    parser.add_argument("--test-size", type=float, default=0.3, help="Fraction for test split")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--stratify-col",
        default=None,
        help="Optional column name for stratified split, e.g. income",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    base_name = input_path.stem

    stratify_values = None
    if args.stratify_col:
        if args.stratify_col not in df.columns:
            raise ValueError(f"Stratify column not found: {args.stratify_col}")
        stratify_values = df[args.stratify_col]

    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        shuffle=True,
        stratify=stratify_values,
    )

    train_path = output_dir / f"{base_name}_train.csv"
    test_path = output_dir / f"{base_name}_test.csv"
    summary_path = output_dir / f"{base_name}_split_summary.txt"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    summary = (
        f"Input rows: {len(df)}\n"
        f"Train rows: {len(train_df)}\n"
        f"Test rows: {len(test_df)}\n"
        f"Test size: {args.test_size}\n"
        f"Random state: {args.random_state}\n"
        f"Stratify column: {args.stratify_col}\n"
    )
    summary_path.write_text(summary, encoding="utf-8")

    print(f"Split complete: {base_name}")
    print(f"Saved: {train_path}")
    print(f"Saved: {test_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()

#python split_data.py --input data\adult_income.csv --output-dir outputs\data --stratify-col income