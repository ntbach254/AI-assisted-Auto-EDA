import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full thesis pipeline with default parameters.")
    parser.add_argument("--input", required=False, help="Path to the raw input CSV")
    parser.add_argument("--output-root", default="outputs", help="Root output directory")
    parser.add_argument("--stratify-col", default=None, help="Optional column for stratified split, e.g. income")
    parser.add_argument("--dataset-name", default=None, help="Optional dataset name override")
    return parser.parse_args()


def infer_dataset_name(input_path: Path, explicit_name: str | None) -> str:
    if explicit_name:
        return explicit_name
    return input_path.stem


def run_step(cmd: list[str]) -> None:
    print("\n" + "=" * 100)
    print("Running:", " ".join(cmd))
    print("=" * 100)

    result = subprocess.run(cmd, text=True, capture_output=True)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Step failed with code {result.returncode}: {' '.join(cmd)}")


def find_single_file(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file found for pattern: {pattern} under {root}")
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple files found for pattern: {pattern} under {root}\n"
            + "\n".join(str(m) for m in matches)
        )
    return matches[0]


def main():
    args = parse_args()

    # GUI fallback for input CSV
    if not args.input:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()

        selected_file = filedialog.askopenfilename(
            title="Select a CSV file",
            filetypes=[("CSV files", "*.csv")]
        )

        if not selected_file:
            print("No file selected.")
            sys.exit(1)

        args.input = selected_file

    base_dir = Path(__file__).resolve().parent
    python_exe = sys.executable

    input_csv = Path(args.input).resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    dataset_name = infer_dataset_name(input_csv, args.dataset_name)

    output_root = Path(args.output_root).resolve()
    dataset_root = output_root / dataset_name

    data_dir = dataset_root / "data"
    preprocess_dir = dataset_root / "preprocess"
    associations_dir = dataset_root / "associations"
    prompt_dir = dataset_root / "prompt"
    hypotheses_dir = dataset_root / "hypotheses"
    validation_dir = dataset_root / "validation"
    summary_dir = dataset_root / "summary"  # NEW

    for d in [data_dir, preprocess_dir, associations_dir, prompt_dir, hypotheses_dir, validation_dir, summary_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Split raw data
    split_cmd = [
        python_exe, str(base_dir / "split_data.py"),
        "--input", str(input_csv),
        "--output-dir", str(data_dir),
    ]
    if args.stratify_col:
        split_cmd.extend(["--stratify-col", args.stratify_col])
    run_step(split_cmd)

    train_csv = data_dir / f"{dataset_name}_train.csv"
    test_csv = data_dir / f"{dataset_name}_test.csv"

    # 2. Preprocess train
    run_step([
        python_exe, str(base_dir / "preprocess.py"),
        "--input", str(train_csv),
        "--output-dir", str(preprocess_dir),
        "--mode", "fit_transform",
    ])

    train_cleaned = preprocess_dir / f"{dataset_name}_train_cleaned.csv"
    train_profile = preprocess_dir / f"{dataset_name}_train_profile.json"
    train_state = preprocess_dir / f"{dataset_name}_train_preprocessing_state.json"

    # 3. Preprocess test using train state
    run_step([
        python_exe, str(base_dir / "preprocess.py"),
        "--input", str(test_csv),
        "--output-dir", str(preprocess_dir),
        "--mode", "transform_only",
        "--state-json", str(train_state),
    ])

    test_cleaned = preprocess_dir / f"{dataset_name}_test_cleaned.csv"

    # 4. Associations on train
    run_step([
        python_exe, str(base_dir / "run_all_associations.py"),
        "--input-csv", str(train_cleaned),
        "--profile-json", str(train_profile),
        "--output-dir", str(associations_dir),
    ])

    # 5. Prompt generation on train associations
    filtered_assoc = find_single_file(associations_dir, f"{dataset_name}_train_associations_filtered.csv")

    run_step([
        python_exe, str(base_dir / "prompt_generate.py"),
        "--profile-json", str(train_profile),
        "--filtered-assoc-csv", str(filtered_assoc),
        "--output-dir", str(prompt_dir),
    ])

    # 6. LLM hypothesis generation
    prompt_file = find_single_file(prompt_dir, f"{dataset_name}_train_hypothesis_prompt.txt")

    run_step([
        python_exe, str(base_dir / "generate_hypotheses.py"),
        "--prompt-file", str(prompt_file),
        "--output-dir", str(hypotheses_dir),
    ])

    # 7. Validation + bootstrap on test
    hypotheses_json = find_single_file(hypotheses_dir, f"{dataset_name}_train_hypotheses.json")

    run_step([
        python_exe, str(base_dir / "validate_hypotheses.py"),
        "--hypotheses-json", str(hypotheses_json),
        "--test-csv", str(test_cleaned),
        "--output-dir", str(validation_dir),
    ])

    final_hypotheses = find_single_file(validation_dir, f"{dataset_name}_train_final_hypotheses.json")
    validation_summary = find_single_file(validation_dir, f"{dataset_name}_train_validation_summary.json")

    # 8. Summarize validated hypotheses (NEW STEP)
    validation_results_csv = find_single_file(
        validation_dir,
        f"{dataset_name}_train_validation_results.csv"
    )

    run_step([
        python_exe, str(base_dir / "summarize_hypotheses.py"),
        "--validation-results", str(validation_results_csv),
        "--output-dir", str(summary_dir),
        "--dataset-name", dataset_name
    ])

    summary_file = summary_dir / f"{dataset_name}_final_summary.txt"

    print("\n" + "#" * 100)
    print("FULL PIPELINE COMPLETE")
    print("#" * 100)
    print(f"Dataset: {dataset_name}")
    print(f"Dataset root: {dataset_root}")
    print(f"Final hypotheses: {final_hypotheses}")
    print(f"Validation summary: {validation_summary}")
    print(f"Final summary: {summary_file}")  # NEW


if __name__ == "__main__":
    main()
