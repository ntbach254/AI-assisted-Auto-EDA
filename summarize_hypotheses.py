import argparse
import subprocess
import json
from pathlib import Path
import pandas as pd
from ollama import chat


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def filter_valid(df: pd.DataFrame) -> pd.DataFrame:
    if "final_accepted" not in df.columns:
        raise ValueError("Missing 'final__accepted' column")

    return df[df["final_accepted"] == True].copy()


def build_structured_input(df: pd.DataFrame) -> str:
    blocks = []

    for i, row in df.iterrows():
        try:
            spec = json.loads(row["validation_spec"])
        except Exception:
            continue

        outcome = spec.get("outcome")
        predictors = spec.get("predictors", [])
        signs = spec.get("expected_signs", {})

        predictor_lines = []
        for p in predictors:
            sign = signs.get(p, "unknown")
            predictor_lines.append(f"- {p}: {sign}")

        block = f"""
[HYPOTHESIS]
Outcome: {outcome}
Predictors:
{chr(10).join(predictor_lines)}

Original Claim:
{row.get("claim", "")}
"""
        blocks.append(block.strip())

    return "\n\n".join(blocks)


def build_prompt(structured_input: str) -> str:
    return f"""
You are given VALIDATED statistical hypotheses.

Each hypothesis contains:
- One outcome variable
- Multiple predictors with expected direction

TASK:
1. Group insights by outcome variable
2. Remove redundancy (duplicate or reversed relationships)
3. Combine predictors into meaningful multi-variable insights
4. Preserve individual predictor effects
5. Interpret clearly and concisely

STRICT RULES:
- Only use relationships present in the input
- Do NOT infer new relationships
- Do NOT convert to causal language
- Ignore unsupported parts of the original claims
- Trust validation_spec over claim text
- Only output the final answer
- Do NOT include thinking, reasoning steps, or analysis
- Do NOT include phrases like "Thinking", "Let's", or internal notes
- Output must be clean, final text only

INPUT:
{structured_input}

OUTPUT FORMAT (PLAIN TEXT):

Group by outcome variable.

For each outcome:
- Start with strongest predictors
- Then describe combined effects
- Keep it concise and non-redundant

Example style:

Wine quality increases with higher alcohol content and decreases with higher density, indicating that lighter, higher-alcohol wines tend to be rated better.
Now produce the final summarized insights:
"""


def call_ollama(prompt: str, model: str) -> str:
    try:
        response = chat(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            options={
                "temperature": 0.2,
                "top_p": 0.9,
            },
        )
    except Exception as e:
        raise RuntimeError(f"Ollama call failed: {e}")

    try:
        content = response["message"]["content"]
    except Exception:
        raise RuntimeError(f"Unexpected Ollama response format: {response}")

    return content.strip()

def clean_output(text: str) -> str:
    return text.replace("\x1b", "").strip()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--validation-results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--model", default="gpt-oss:120b-cloud")

    args = parser.parse_args()

    input_path = Path(args.validation_results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(input_path)
    df_valid = filter_valid(df)

    if df_valid.empty:
        raise ValueError("No accepted hypotheses found")

    structured_input = build_structured_input(df_valid)
    prompt = build_prompt(structured_input)

    summary_raw = call_ollama(prompt, args.model)

    print("\n[DEBUG] RAW OUTPUT:\n")
    print(summary_raw[:1000])  # preview first 1000 chars

    summary = clean_output(summary_raw)

    output_file = output_dir / f"{args.dataset_name}_final_summary.md"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"[OK] Summary saved to: {output_file}")


if __name__ == "__main__":
    main()

#python summarize_hypotheses.py --validation-results "D:\THESIS\outputs\wine_quality_red\validation\wine_quality_red_train_validation_results.csv" --output-dir "D:\THESIS\outputs\wine_quality_red\summary" --dataset-name wine_quality_red
