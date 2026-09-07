import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class EvidencePackConfig:
    max_edges: int = 12
    max_vars: int = 40
    max_json_chars: int = 20000

    drop_weight_like_vars: bool = True
    drop_near_deterministic_edges: bool = True
    near_deterministic_threshold: float = 0.95

    max_edges_per_variable: int = 3

    include_profile_fields: Tuple[str, ...] = (
        "feature_type",
        "dtype",
        "missing_rate",
        "n_unique",
        "skewness",
        "outlier_rate",
        "issues",
    )


class EvidencePackBuilder:
    def __init__(self, config: Optional[EvidencePackConfig] = None):
        self.cfg = config or EvidencePackConfig()

    def _is_weight_like(self, var_name: str) -> bool:
        name = str(var_name).lower()
        return ("weight" in name) or ("fnlwgt" in name)

    def _pick_profile(self, profile_item: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k in self.cfg.include_profile_fields:
            if k in profile_item:
                out[k] = profile_item[k]
        return out

    def _effect_bucket(self, assoc_type: str, effect: float) -> str:
        x = abs(float(effect))

        if assoc_type == "num_num":
            if x >= 0.50:
                return "large"
            if x >= 0.30:
                return "medium"
            if x >= 0.10:
                return "small"
            return "tiny"

        if assoc_type == "cat_cat":
            if x >= 0.50:
                return "large"
            if x >= 0.30:
                return "medium"
            if x >= 0.15:
                return "small"
            return "tiny"

        if x >= 0.14:
            return "large"
        if x >= 0.06:
            return "medium"
        if x >= 0.03:
            return "small"
        return "tiny"

    def _variance_like_value(self, assoc_type: str, effect: float, effect_name: str):
        x = float(effect)

        if assoc_type == "num_num":
            return round(x * x, 6)

        if assoc_type == "num_cat" and effect_name == "eta2_approx":
            return round(x, 6)

        return None

    def _constraints_for_edge(self, row: Dict[str, Any]) -> List[str]:
        constraints = []

        assoc_type = row["type"]
        effect = abs(float(row["effect"]))
        bucket = self._effect_bucket(assoc_type, effect)

        if bucket in {"tiny", "small"}:
            constraints.append("effect_is_weak_or_modest")
            constraints.append("do_not_overstate_strength")

        if assoc_type == "num_num" and row["test"] == "spearmanr":
            constraints.append("monotonic_not_necessarily_linear")
        elif assoc_type == "num_cat":
            constraints.append("interpret_as_group_difference")
        elif assoc_type == "cat_cat":
            constraints.append("interpret_as_category_association")

        notes = str(row.get("notes", "") or "")
        if "sparse_table" in notes:
            constraints.append("chi_square_table_may_be_sparse")
        if "nonparametric" in notes:
            constraints.append("nonparametric_test_used")

        return constraints

    def _edge_priority(self, row: pd.Series) -> float:
        effect = abs(float(row["effect"]))
        assoc_type = row["type"]
        score = effect

        if assoc_type == "num_num":
            score += 0.06
        elif assoc_type == "num_cat":
            score += 0.04
        elif assoc_type == "cat_cat":
            score += 0.03

        notes = str(row.get("notes", "") or "")
        if "nonparametric" in notes:
            score += 0.01
        if "sparse_table" in notes:
            score -= 0.15

        if self._effect_bucket(assoc_type, effect) == "small":
            score -= 0.03

        if self._is_weight_like(row["var1"]) or self._is_weight_like(row["var2"]):
            score -= 1.0

        if effect >= self.cfg.near_deterministic_threshold:
            score -= 0.5

        return score

    def _pair_key(self, row: pd.Series) -> Tuple[str, str]:
        return tuple(sorted([str(row["var1"]), str(row["var2"])]))

    def _select_top_edges(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) == 0:
            return df.copy()

        df = df.copy()
        df["pair_key"] = df.apply(self._pair_key, axis=1)
        df = df.drop_duplicates("pair_key").copy()

        df["priority_score"] = df.apply(self._edge_priority, axis=1)
        df = df.sort_values(
            ["priority_score", "adjusted_p", "effect"],
            ascending=[False, True, False]
        ).reset_index(drop=True)

        selected_rows = []
        var_counts: Dict[str, int] = {}
        pair_seen = set()

        for _, row in df.iterrows():
            v1 = str(row["var1"])
            v2 = str(row["var2"])
            pair_key = row["pair_key"]

            if pair_key in pair_seen:
                continue

            if var_counts.get(v1, 0) >= self.cfg.max_edges_per_variable:
                continue
            if var_counts.get(v2, 0) >= self.cfg.max_edges_per_variable:
                continue

            selected_rows.append(row)
            pair_seen.add(pair_key)
            var_counts[v1] = var_counts.get(v1, 0) + 1
            var_counts[v2] = var_counts.get(v2, 0) + 1

            if len(selected_rows) >= self.cfg.max_edges:
                break

        selected = pd.DataFrame(selected_rows).reset_index(drop=True)

        if len(selected) < min(6, self.cfg.max_edges):
            selected_pairs = set(selected["pair_key"]) if "pair_key" in selected.columns and len(selected) > 0 else set()
            extra = []
            for _, row in df.iterrows():
                if row["pair_key"] in selected_pairs:
                    continue
                extra.append(row)
                if len(selected) + len(extra) >= self.cfg.max_edges:
                    break
            if extra:
                selected = pd.concat([selected, pd.DataFrame(extra)], ignore_index=True)

        for col in ["pair_key", "priority_score"]:
            if col in selected.columns:
                selected = selected.drop(columns=[col])

        return selected

    def _trim_to_budget(self, pack: Dict[str, Any]) -> Dict[str, Any]:
        s = json.dumps(pack, ensure_ascii=False)
        if len(s) <= self.cfg.max_json_chars:
            return pack

        edges = pack.get("edges", [])
        lo, hi = 0, len(edges)
        best = 0

        while lo <= hi:
            mid = (lo + hi) // 2
            test_pack = dict(pack)
            test_pack["edges"] = edges[:mid]
            test_s = json.dumps(test_pack, ensure_ascii=False)

            if len(test_s) <= self.cfg.max_json_chars:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1

        pack["edges"] = edges[:best]

        used_vars = set()
        for e in pack["edges"]:
            used_vars.add(e["var1"])
            used_vars.add(e["var2"])

        old_vars = pack.get("variables", {})
        pack["variables"] = {k: v for k, v in old_vars.items() if k in used_vars}

        return pack

    def build(
        self,
        profile: Dict[str, Dict[str, Any]],
        filtered_assoc: pd.DataFrame,
        dataset_name: str = "dataset",
        notes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        notes = notes or []

        required_cols = {
            "var1", "var2", "type", "relationship", "test",
            "effect", "effect_name", "p_value", "adjusted_p", "n"
        }
        missing = required_cols - set(filtered_assoc.columns)
        if missing:
            raise ValueError(f"filtered_assoc missing columns: {sorted(missing)}")

        df = filtered_assoc.copy()

        if self.cfg.drop_weight_like_vars:
            df = df[
                ~df["var1"].apply(self._is_weight_like) &
                ~df["var2"].apply(self._is_weight_like)
            ].copy()

        if self.cfg.drop_near_deterministic_edges:
            df = df[df["effect"].abs() < self.cfg.near_deterministic_threshold].copy()

        if len(df) == 0:
            return {
                "schema_version": "2.3",
                "dataset": dataset_name,
                "summary": {
                    "n_edges": 0,
                    "n_variables": 0,
                    "message": "No associations remained after final evidence-pack pruning."
                },
                "variables": {},
                "edges": [],
                "global_constraints": [
                    "do_not_infer_causality",
                    "do_not_use_external_facts",
                    "return_only_testable_hypotheses"
                ],
                "notes": notes,
            }

        df = self._select_top_edges(df)

        vars_in_edges = sorted(set(df["var1"]).union(set(df["var2"])))
        vars_in_edges = vars_in_edges[: self.cfg.max_vars]

        variables_block = {
            v: self._pick_profile(profile.get(v, {}))
            for v in vars_in_edges
        }

        edges_block = []
        for i, row in enumerate(df.to_dict(orient="records"), start=1):
            assoc_type = row["type"]
            effect = float(row["effect"])
            bucket = self._effect_bucket(assoc_type, effect)
            var_like = self._variance_like_value(assoc_type, effect, row["effect_name"])

            edge = {
                "edge_id": f"E{i}",
                "var1": row["var1"],
                "var2": row["var2"],
                "type": assoc_type,
                "relationship": row["relationship"],
                "test": row["test"],
                "effect": round(effect, 6),
                "effect_name": row["effect_name"],
                "effect_size": bucket,
                "n": int(row["n"]),
                "q_value": float(row["adjusted_p"]),
                "notes": str(row.get("notes", "") or ""),
            }

            constraints = self._constraints_for_edge(row)
            if constraints:
                edge["inference_constraints"] = constraints

            if assoc_type == "num_num":
                edge["direction"] = "positive" if effect > 0 else "negative"
            elif assoc_type == "num_cat":
                if row["effect_name"] == "rank_biserial":
                    edge["direction"] = "group_difference_direction_depends_on_category_order"
                else:
                    edge["direction"] = "group_difference"
            else:
                edge["direction"] = "association_without_direction"

            if var_like is not None:
                edge["variance_explained_approx"] = var_like

            edges_block.append(edge)

        pack = {
            "schema_version": "2.3",
            "dataset": dataset_name,
            "summary": {
                "n_edges": len(edges_block),
                "n_variables": len(variables_block),
                "association_types_present": sorted(df["type"].unique().tolist()),
                "tests_present": sorted(df["test"].unique().tolist()),
            },
            "variables": variables_block,
            "edges": edges_block,
            "global_constraints": [
                "do_not_infer_causality",
                "do_not_use_external_facts",
                "do_not_overstate_effect_sizes",
                "prefer_non_redundant_hypotheses",
                "prefer_hypotheses_that_synthesize_multiple_edges_when_supported",
                "every_hypothesis_must_reference_at_least_one_edge",
                "every_hypothesis_must_include_a_test_plan",
            ],
            "notes": notes,
        }

        return self._trim_to_budget(pack)

    def _compute_hypothesis_count(
        self,
        n_edges: int,
        requested_hypotheses: Optional[int] = None
    ) -> int:
        auto_n = min(18, max(15, int(round(n_edges * 1.4))))

        if requested_hypotheses is None:
            return auto_n

        return min(18, max(15, min(requested_hypotheses, auto_n + 2)))

    def build_prompt(
        self,
        evidence_pack: Dict[str, Any],
        requested_hypotheses: Optional[int] = None,
    ) -> str:
        n_edges = len(evidence_pack.get("edges", []))
        n_hypotheses = self._compute_hypothesis_count(n_edges, requested_hypotheses)

        output_schema = {
            "hypotheses": [
                {
                    "id": "H1",
                    "variables": ["income", "education_num", "age"],
                    "relationship_type": "association|group_difference|monotonic|category_association|multi_edge_pattern",
                    "direction": "positive|negative|group_difference|association_without_direction|unknown",
                    "claim": "One sentence grounded only in the evidence pack.",
                    "supporting_evidence": ["E1", "E4"],
                    "test_plan": {
                        "primary_test": "spearmanr|mannwhitneyu|kruskal|chi2|pearsonr|logistic_regression|linear_regression",
                        "null_hypothesis": "string",
                        "alpha": 0.05,
                        "multiple_testing": "BH over this batch"
                    },
                    "validation_spec": {
                        "outcome": "income",
                        "predictors": ["education_num"],
                        "controls": ["age"],
                        "expected_signs": {
                            "education_num": "positive",
                            "age": "unknown"
                        },
                        "analysis_unit": "row",
                        "model_formula": "income ~ education_num + age"
                    },
                    "risk_flags": ["weak_effect", "possible_confounding", "nonparametric_result", "redundancy_risk"]
                }
            ]
        }

        pack_json = json.dumps(evidence_pack, ensure_ascii=False, indent=2)
        schema_json = json.dumps(output_schema, ensure_ascii=False, indent=2)

        prompt = f"""
You are generating testable hypotheses from a structured evidence pack.

Task:
Generate {n_hypotheses} high-value, non-redundant, testable hypotheses grounded only in the evidence pack.

Core objective:
Produce hypotheses that reveal meaningful structure in the data, not simple restatements of associations.

Rules:
1. Use only the evidence_pack. Do not introduce external knowledge.
2. Do not claim causality. Only propose association, group difference, monotonic, category association, or multi-edge pattern hypotheses.
3. Avoid trivial hypotheses:
   - Do not restate a single edge in plain words.
   - Do not merely say that several variables are associated.
4. Enforce real synthesis:
   - At least half of the hypotheses should combine multiple edges when the evidence pack supports it.
   - Multi-edge hypotheses must express a specific testable structure such as shared association patterns, conditional relationships, or joint variation.
   - Do not simply merge multiple edges into a list of variables.
5. Avoid redundancy:
   - Do not generate two hypotheses that use highly overlapping variable sets unless the claimed pattern is clearly different.
   - Each hypothesis should introduce a meaningfully different pattern.
6. Prefer informative hypotheses:
   - Avoid repeatedly focusing on the same small categorical cluster unless the structure is clearly different.
   - Prefer hypotheses that connect different parts of the graph when supported.
7. Respect effect sizes:
   - If the evidence is weak or modest, state the claim cautiously.
   - Do not exaggerate weak effects.
8. Use an appropriate statistical test that matches the variables involved.
9. Every hypothesis must be machine-testable from the output JSON alone.
10. Every hypothesis must reference at least one edge_id from the evidence_pack.
11. Use edge_ids explicitly in "supporting_evidence".
12. Every hypothesis must include a validation_spec.
13. In validation_spec, specify exactly:
   - one explicit outcome variable
   - one or more predictors
   - optional controls
   - expected sign for each predictor when applicable
   - a model_formula string
14. For pairwise hypotheses, validation_spec must still be included, with outcome and predictor(s) matching the pairwise test.
15. For multi-variable hypotheses, do not invent causal mechanisms or unjustified adjustment sets.
16. Controls must be minimal and grounded in the evidence pack.
17. If a multi-variable hypothesis cannot be expressed with a clear outcome and predictors, do not generate it.
18. Return ONLY valid JSON matching the schema exactly. Do not include explanations, comments, markdown fences, or any text outside the JSON object.
19. Every variable named in variables, validation_spec.outcome, validation_spec.predictors, and validation_spec.controls must exactly match a variable name from the evidence_pack.
20. Do not use any variable that is not present in evidence_pack.variables.
21. logistic_regression may only be used when the outcome variable is binary.
22. linear_regression may only be used when the outcome variable is numeric_continuous or numeric_ordinal.
23. chi2 may only be used when both outcome and predictor are categorical or binary.
24. mannwhitneyu may only be used when the outcome is numeric and the predictor has exactly 2 groups.
25. kruskal may only be used when the outcome is numeric and the predictor is categorical with 3 or more groups.
26. pearsonr and spearmanr may only be used when both variables are numeric.
27. If you cannot produce a fully testable hypothesis under these rules, do not output that hypothesis.
28. Prefer fewer hypotheses over invalid hypotheses.
29. Only use linear_regression or logistic_regression when the evidence pack strongly supports a multi-variable hypothesis and the outcome type is clearly compatible.
30. Prefer pairwise tests unless a multi-variable regression hypothesis is clearly justified by multiple edges.
evidence_pack:
{pack_json}

output_schema:
{schema_json}

For each hypothesis, you MUST follow this two-part structure:

1. Hypothesis Statement (Formal)
- State the relationship clearly and precisely using statistical language.
- Specify variables and direction (positive, negative, monotonic, etc.) when applicable.
- Avoid vague phrases like “is related to” unless no direction can be inferred.

2. Interpretation (Insight)
- Immediately explain what the relationship means in practical or real-world terms.
- Clarify how the variables interact and why this pattern might occur.
- If multiple variables are involved, explain their combined effect, not just individually.
- Highlight any notable pattern (e.g., amplification, trade-off, interaction effect).

Formatting:
- Combine both parts into ONE coherent paragraph.
- Start with the formal hypothesis, then follow with the interpretation.
- Do NOT repeat the same idea in different words.
- Do NOT produce separate bullet points.

Quality constraints:
- Avoid generic or obvious statements.
- Prefer specific, data-driven insights over safe descriptions.
- If multiple related hypotheses overlap, consolidate them into a single richer explanation.

Return ONLY valid JSON. Do not include explanations, comments, or any text outside the JSON object.
""".strip()

        return prompt


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-json", required=True)
    parser.add_argument("--filtered-assoc-csv", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--max-edges", type=int, default=12)
    parser.add_argument("--max-vars", type=int, default=30)
    parser.add_argument("--max-json-chars", type=int, default=18000)
    parser.add_argument("--near-deterministic-threshold", type=float, default=0.95)
    parser.add_argument("--max-edges-per-variable", type=int, default=3)

    parser.add_argument("--requested-hypotheses", type=int, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    profile_path = Path(args.profile_json)
    filtered_assoc_path = Path(args.filtered_assoc_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = filtered_assoc_path.stem.replace("_associations_filtered", "")

    profile = load_json(profile_path)
    filtered_assoc = pd.read_csv(filtered_assoc_path)

    builder = EvidencePackBuilder(
        EvidencePackConfig(
            max_edges=args.max_edges,
            max_vars=args.max_vars,
            max_json_chars=args.max_json_chars,
            drop_weight_like_vars=True,
            drop_near_deterministic_edges=True,
            near_deterministic_threshold=args.near_deterministic_threshold,
            max_edges_per_variable=args.max_edges_per_variable,
        )
    )

    evidence_pack = builder.build(
        profile=profile,
        filtered_assoc=filtered_assoc,
        dataset_name=dataset_name,
        notes=[
            "Top edges were selected with a per-variable cap.",
            "Weight-like variables were excluded.",
            "Near-deterministic edges were excluded."
        ]
    )

    prompt = builder.build_prompt(
        evidence_pack=evidence_pack,
        requested_hypotheses=args.requested_hypotheses
    )

    evidence_pack_path = output_dir / f"{dataset_name}_evidence_pack.json"
    prompt_path = output_dir / f"{dataset_name}_hypothesis_prompt.txt"
    metadata_path = output_dir / f"{dataset_name}_prompt_metadata.json"

    save_json(evidence_pack, evidence_pack_path)
    prompt_path.write_text(prompt, encoding="utf-8")

    metadata = {
        "dataset": dataset_name,
        "n_edges_in_pack": len(evidence_pack.get("edges", [])),
        "n_variables_in_pack": len(evidence_pack.get("variables", {})),
        "requested_hypotheses": args.requested_hypotheses,
        "prompt_char_count": len(prompt),
        "config": {
            "max_edges": args.max_edges,
            "max_vars": args.max_vars,
            "max_json_chars": args.max_json_chars,
            "near_deterministic_threshold": args.near_deterministic_threshold,
            "max_edges_per_variable": args.max_edges_per_variable,
        }
    }
    save_json(metadata, metadata_path)

    print(f"Prompt generation complete: {dataset_name}")
    print(f"Saved: {evidence_pack_path}")
    print(f"Saved: {prompt_path}")
    print(f"Saved: {metadata_path}")


if __name__ == "__main__":
    main()

#python prompt_generate.py --profile-json outputs\preprocess\adult_income_train_profile.json --filtered-assoc-csv outputs\associations\adult_income_train_associations_filtered.csv --output-dir outputs\prompt