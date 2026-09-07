import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--profile-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-n", type=int, default=30)
    parser.add_argument("--max-categories", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-abs-r", type=float, default=0.15)
    parser.add_argument("--min-cramers-v", type=float, default=0.20)
    parser.add_argument("--min-eta2", type=float, default=0.05)
    parser.add_argument("--min-rank-biserial", type=float, default=0.15)
    return parser.parse_args()


def run_step(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()

    input_csv = Path(args.input_csv)
    profile_json = Path(args.profile_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = input_csv.stem.replace("_cleaned", "")

    python_exe = sys.executable

    run_step([
        python_exe, "num_num.py",
        "--input-csv", str(input_csv),
        "--profile-json", str(profile_json),
        "--output-dir", str(output_dir),
        "--min-n", str(args.min_n),
    ])

    run_step([
        python_exe, "num_cat.py",
        "--input-csv", str(input_csv),
        "--profile-json", str(profile_json),
        "--output-dir", str(output_dir),
        "--min-n", str(args.min_n),
        "--max-categories", str(args.max_categories),
    ])

    run_step([
        python_exe, "cat_cat.py",
        "--input-csv", str(input_csv),
        "--profile-json", str(profile_json),
        "--output-dir", str(output_dir),
        "--min-n", str(args.min_n),
        "--max-categories", str(args.max_categories),
    ])

    run_step([
        python_exe, "fdr_correction.py",
        "--input-dir", str(output_dir),
        "--dataset-name", dataset_name,
        "--output-dir", str(output_dir),
        "--alpha", str(args.alpha),
        "--min-n", str(args.min_n),
        "--min-abs-r", str(args.min_abs_r),
        "--min-cramers-v", str(args.min_cramers_v),
        "--min-eta2", str(args.min_eta2),
        "--min-rank-biserial", str(args.min_rank_biserial),
    ])

    print(f"Association pipeline complete for: {dataset_name}")


if __name__ == "__main__":
    main()
#python run_all_associations.py --input-csv outputs\preprocess\adult_income_train_cleaned.csv --profile-json outputs\preprocess\adult_income_train_profile.json --output-dir outputs\associations --min-abs-r 0.15 --min-cramers-v 0.20 --min-eta2 0.05 --min-rank-biserial 0.15