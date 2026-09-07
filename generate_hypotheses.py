import argparse
import json
import sys
from pathlib import Path

from ollama import chat


DEFAULT_MODEL = "gpt-oss:120b-cloud"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", required=True, help="Path to the prompt .txt file")
    parser.add_argument("--output-dir", required=True, help="Directory to save outputs")
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Optional dataset name. If omitted, inferred from prompt filename.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    return parser.parse_args()


def infer_dataset_name(prompt_file: Path, explicit_name: str | None) -> str:
    if explicit_name:
        return explicit_name

    stem = prompt_file.stem
    known_suffixes = [
        "_hypothesis_prompt",
        "_prompt",
    ]

    for suffix in known_suffixes:
        if stem.endswith(suffix):
            return stem[:-len(suffix)]

    return stem


def extract_json(text: str) -> dict:
    text = text.strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj

    raise ValueError("Model output is not valid JSON.")


def validate_schema(data: dict) -> None:
    if "hypotheses" not in data or not isinstance(data["hypotheses"], list):
        raise ValueError("Output must contain a 'hypotheses' list.")

    required = {
        "id",
        "variables",
        "relationship_type",
        "direction",
        "claim",
        "supporting_evidence",
        "test_plan",
        "risk_flags",
    }

    test_plan_required = {
        "primary_test",
        "null_hypothesis",
        "alpha",
        "multiple_testing",
    }

    for i, h in enumerate(data["hypotheses"], start=1):
        if not isinstance(h, dict):
            raise ValueError(f"Hypothesis {i} is not an object.")

        missing = required - set(h.keys())
        if missing:
            raise ValueError(f"Hypothesis {i} missing keys: {sorted(missing)}")

        if not isinstance(h["variables"], list):
            raise ValueError(f"Hypothesis {i} 'variables' must be a list.")

        if not isinstance(h["supporting_evidence"], list):
            raise ValueError(f"Hypothesis {i} 'supporting_evidence' must be a list.")

        if not isinstance(h["risk_flags"], list):
            raise ValueError(f"Hypothesis {i} 'risk_flags' must be a list.")

        if not isinstance(h["test_plan"], dict):
            raise ValueError(f"Hypothesis {i} test_plan must be an object.")

        missing_tp = test_plan_required - set(h["test_plan"].keys())
        if missing_tp:
            raise ValueError(f"Hypothesis {i} test_plan missing keys: {sorted(missing_tp)}")


def save_json(obj: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()

    prompt_file = Path(args.prompt_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not prompt_file.exists():
        print(f"Prompt file not found: {prompt_file}", file=sys.stderr)
        sys.exit(1)

    dataset_name = infer_dataset_name(prompt_file, args.dataset_name)
    prompt = prompt_file.read_text(encoding="utf-8")

    try:
        response = chat(
            model=args.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            options={
                "temperature": args.temperature,
                "top_p": 0.9,
            },
        )
    except Exception as e:
        print(f"Ollama call failed: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        content = response["message"]["content"]
    except Exception:
        print("Unexpected Ollama response format.", file=sys.stderr)
        print(response, file=sys.stderr)
        sys.exit(1)

    raw_path = output_dir / f"{dataset_name}_hypothesis_raw.txt"
    parsed_path = output_dir / f"{dataset_name}_hypotheses.json"
    metadata_path = output_dir / f"{dataset_name}_hypothesis_metadata.json"

    raw_path.write_text(content, encoding="utf-8")

    try:
        parsed = extract_json(content)
        validate_schema(parsed)
    except Exception as e:
        error_path = output_dir / f"{dataset_name}_hypothesis_error.txt"
        error_path.write_text(
            f"{type(e).__name__}: {e}\n\nRAW OUTPUT:\n{content}",
            encoding="utf-8"
        )
        print(f"Invalid model output: {e}", file=sys.stderr)
        print(f"Saved error details to: {error_path}", file=sys.stderr)
        sys.exit(2)

    save_json(parsed, parsed_path)

    metadata = {
        "dataset": dataset_name,
        "model": args.model,
        "temperature": args.temperature,
        "prompt_file": str(prompt_file),
        "raw_output_file": str(raw_path),
        "parsed_output_file": str(parsed_path),
        "hypothesis_count": len(parsed.get("hypotheses", [])),
    }
    save_json(metadata, metadata_path)

    print(f"Done: {dataset_name}")
    print(f"Saved: {raw_path}")
    print(f"Saved: {parsed_path}")
    print(f"Saved: {metadata_path}")


if __name__ == "__main__":
    main()

#python generate_hypotheses.py --prompt-file outputs\prompt\adult_income_train_hypothesis_prompt.txt --output-dir outputs\hypotheses