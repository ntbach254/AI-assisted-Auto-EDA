import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from bh_fdr import benjamini_hochberg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypotheses-json", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default=None)

    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-n", type=int, default=30)

    parser.add_argument("--min-abs-r", type=float, default=0.15)
    parser.add_argument("--min-cramers-v", type=float, default=0.20)
    parser.add_argument("--min-eta2", type=float, default=0.05)
    parser.add_argument("--min-rank-biserial", type=float, default=0.15)

    parser.add_argument("--bootstrap-iterations", type=int, default=200)
    parser.add_argument("--bootstrap-random-state", type=int, default=42)
    parser.add_argument("--bootstrap-sign-consistency-min", type=float, default=0.80)
    parser.add_argument("--bootstrap-effect-pass-rate-min", type=float, default=0.70)

    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def infer_dataset_name(hypotheses_path: Path, explicit_name: str | None) -> str:
    if explicit_name:
        return explicit_name

    stem = hypotheses_path.stem
    known_suffixes = [
        "_hypotheses",
        "_train_hypotheses",
    ]
    for suffix in known_suffixes:
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def eta_squared(groups):
    all_vals = np.concatenate(groups)
    grand_mean = np.mean(all_vals)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_total = sum((v - grand_mean) ** 2 for v in all_vals)
    if ss_total <= 0:
        return 0.0
    return float(ss_between / ss_total)


def rank_biserial(a, b):
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return 0.0, 1.0
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    rbc = 1.0 - (2.0 * u) / (na * nb)
    return float(rbc), float(p)


def cramers_v(contingency):
    chi2, p, _, _ = stats.chi2_contingency(contingency, correction=False)
    n = contingency.to_numpy().sum()
    r, c = contingency.shape
    denom = n * min(r - 1, c - 1)
    v = 0.0 if denom <= 0 else float(np.sqrt(chi2 / denom))
    return float(v), float(p)


def safe_direction_match(expected_direction: str, observed_effect: float, test_name: str):
    expected = str(expected_direction or "").strip()

    if expected in {"", "unknown", "association_without_direction", "group_difference"}:
        return True

    if test_name in {"pearsonr", "spearmanr", "linear_regression", "logistic_regression"}:
        if expected == "positive":
            return observed_effect > 0
        if expected == "negative":
            return observed_effect < 0

    return True


def effect_threshold_pass(effect_name: str, effect_value: float, args) -> bool:
    x = float(effect_value)

    if effect_name in {"r", "rho", "coef"}:
        return abs(x) >= args.min_abs_r
    if effect_name == "cramers_v":
        return x >= args.min_cramers_v
    if effect_name == "eta2_approx":
        return x >= args.min_eta2
    if effect_name == "rank_biserial":
        return abs(x) >= args.min_rank_biserial

    return False


def to_numeric_or_dummy(series: pd.Series, prefix: str) -> pd.DataFrame:
    if pd.api.types.is_numeric_dtype(series):
        out = pd.DataFrame({prefix: pd.to_numeric(series, errors="coerce")})
        return out.astype(float)

    s = series.astype(str)
    dummies = pd.get_dummies(s, prefix=prefix, drop_first=True)

    if dummies.shape[1] == 0:
        return pd.DataFrame(index=series.index)

    return dummies.astype(float)


def build_model_matrix(df: pd.DataFrame, outcome: str, predictors: list[str], controls: list[str]):
    required_cols = [outcome] + predictors + controls
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        return None, None, f"missing_columns:{missing_cols}"

    work = df[required_cols].copy()

    y_raw = work[outcome]
    X_parts = []

    for col in predictors + controls:
        X_part = to_numeric_or_dummy(work[col], col)
        X_parts.append(X_part)

    if not X_parts:
        return None, None, "no_predictors"

    X = pd.concat(X_parts, axis=1)

    if X.shape[1] == 0:
        return None, None, "empty_design_matrix"

    X = X.apply(pd.to_numeric, errors="coerce").astype(float)

    if pd.api.types.is_numeric_dtype(y_raw):
        y = pd.to_numeric(y_raw, errors="coerce").astype(float)
        y_type = "numeric"
    else:
        y_cat = y_raw.astype(str)
        uniq = sorted(y_cat.dropna().unique())
        if len(uniq) != 2:
            return None, None, "non_numeric_outcome_not_binary"
        mapping = {uniq[0]: 0, uniq[1]: 1}
        y = y_cat.map(mapping).astype(float)
        y_type = "binary"

    joined = pd.concat([y.rename("__y__"), X], axis=1).dropna()
    if len(joined) == 0:
        return None, None, "no_complete_cases"

    joined = joined.apply(pd.to_numeric, errors="coerce").dropna()

    y = joined["__y__"].astype(float)
    X = joined.drop(columns=["__y__"]).astype(float)

    if X.shape[1] == 0:
        return None, None, "empty_design_matrix"

    return (y, X, y_type), joined, ""


def post_filter_hypotheses(hypotheses: list[dict], df_test: pd.DataFrame):
    cleaned = []
    rejected_rows = []
    available_columns = set(df_test.columns)

    allowed_tests = {
        "pearsonr",
        "spearmanr",
        "mannwhitneyu",
        "kruskal",
        "chi2",
        "linear_regression",
        "logistic_regression",
    }

    for h in hypotheses:
        h_id = h.get("id", "")
        variables = h.get("variables", [])
        primary_test = h.get("test_plan", {}).get("primary_test", "")
        validation_spec = h.get("validation_spec", {})

        reason = None

        if not isinstance(h, dict):
            reason = "hypothesis_not_object"
        elif not isinstance(variables, list) or len(variables) < 2:
            reason = "invalid_or_too_short_variables"
        elif primary_test not in allowed_tests:
            reason = f"unsupported_primary_test:{primary_test}"
        elif not isinstance(validation_spec, dict) or not validation_spec:
            reason = "missing_validation_spec"
        else:
            outcome = validation_spec.get("outcome")
            predictors = validation_spec.get("predictors", [])
            controls = validation_spec.get("controls", [])

            if not outcome:
                reason = "validation_spec_missing_outcome"
            elif not isinstance(predictors, list) or len(predictors) == 0:
                reason = "validation_spec_missing_predictors"
            elif not isinstance(controls, list):
                reason = "validation_spec_controls_not_list"
            else:
                named_vars = {outcome, *predictors, *controls}
                missing_vars = sorted(v for v in named_vars if v not in available_columns)
                if missing_vars:
                    reason = f"validation_spec_variables_missing_in_test_data:{missing_vars}"
                elif primary_test in {"pearsonr", "spearmanr", "mannwhitneyu", "kruskal", "chi2"} and len(predictors) != 1:
                    reason = "pairwise_test_requires_exactly_one_predictor"
                elif primary_test == "logistic_regression":
                    outcome_series = df_test[outcome]
                    if pd.api.types.is_numeric_dtype(outcome_series):
                        uniq = sorted(pd.Series(outcome_series).dropna().unique())
                        if len(uniq) != 2:
                            reason = "logistic_regression_outcome_not_binary"
                    else:
                        uniq = sorted(outcome_series.astype(str).dropna().unique())
                        if len(uniq) != 2:
                            reason = "logistic_regression_outcome_not_binary"
                elif primary_test == "linear_regression":
                    if not pd.api.types.is_numeric_dtype(df_test[outcome]):
                        reason = "linear_regression_outcome_not_numeric"

        if reason is None:
            cleaned.append(h)
        else:
            rejected_rows.append({
                "id": h_id,
                "primary_test": primary_test,
                "reason": reason,
            })

    return cleaned, pd.DataFrame(rejected_rows)


def run_pairwise_test(df: pd.DataFrame, validation_spec: dict, primary_test: str):
    outcome = validation_spec.get("outcome")
    predictors = validation_spec.get("predictors", [])
    if not outcome or not predictors or len(predictors) != 1:
        return {
            "testable": False,
            "notes": "pairwise_test_requires_exactly_one_predictor_and_one_outcome",
        }

    predictor = predictors[0]
    if outcome not in df.columns or predictor not in df.columns:
        return {
            "testable": False,
            "notes": "missing_variables_in_test_data",
        }

    pair = pd.concat([df[outcome], df[predictor]], axis=1).dropna()
    n = int(len(pair))
    if n == 0:
        return {
            "testable": False,
            "notes": "no_complete_cases",
        }

    y = pair.iloc[:, 0]
    x = pair.iloc[:, 1]

    if primary_test == "pearsonr":
        y = pd.to_numeric(y, errors="coerce")
        x = pd.to_numeric(x, errors="coerce")
        pair2 = pd.concat([y, x], axis=1).dropna()
        if len(pair2) < 2 or pair2.iloc[:, 0].nunique() < 2 or pair2.iloc[:, 1].nunique() < 2:
            return {"testable": False, "notes": "insufficient_variation_for_pearson"}
        effect, p = stats.pearsonr(pair2.iloc[:, 0], pair2.iloc[:, 1])
        return {
            "testable": True,
            "n": int(len(pair2)),
            "observed_test": "pearsonr",
            "effect": float(effect),
            "effect_name": "r",
            "p_value": float(p),
            "notes": "",
        }

    if primary_test == "spearmanr":
        y = pd.to_numeric(y, errors="coerce")
        x = pd.to_numeric(x, errors="coerce")
        pair2 = pd.concat([y, x], axis=1).dropna()
        if len(pair2) < 2 or pair2.iloc[:, 0].nunique() < 2 or pair2.iloc[:, 1].nunique() < 2:
            return {"testable": False, "notes": "insufficient_variation_for_spearman"}
        effect, p = stats.spearmanr(pair2.iloc[:, 0], pair2.iloc[:, 1])
        return {
            "testable": True,
            "n": int(len(pair2)),
            "observed_test": "spearmanr",
            "effect": float(effect),
            "effect_name": "rho",
            "p_value": float(p),
            "notes": "",
        }

    if primary_test == "mannwhitneyu":
        y_num = pd.to_numeric(y, errors="coerce")
        g = x.astype(str)
        pair2 = pd.concat([y_num, g], axis=1).dropna()
        if pair2.iloc[:, 1].nunique() != 2:
            return {"testable": False, "notes": "mannwhitneyu_requires_exactly_2_groups"}
        levels = sorted(pair2.iloc[:, 1].unique())
        a = pair2.loc[pair2.iloc[:, 1] == levels[0], pair2.columns[0]].to_numpy()
        b = pair2.loc[pair2.iloc[:, 1] == levels[1], pair2.columns[0]].to_numpy()
        if len(a) < 2 or len(b) < 2:
            return {"testable": False, "notes": "mannwhitneyu_groups_too_small"}
        effect, p = rank_biserial(a, b)
        return {
            "testable": True,
            "n": int(len(pair2)),
            "observed_test": "mannwhitneyu",
            "effect": float(effect),
            "effect_name": "rank_biserial",
            "p_value": float(p),
            "notes": "",
        }

    if primary_test == "kruskal":
        y_num = pd.to_numeric(y, errors="coerce")
        g = x.astype(str)
        pair2 = pd.concat([y_num, g], axis=1).dropna()
        levels = sorted(pair2.iloc[:, 1].unique())
        if len(levels) < 2:
            return {"testable": False, "notes": "kruskal_requires_2_or_more_groups"}
        groups = [
            pair2.loc[pair2.iloc[:, 1] == lvl, pair2.columns[0]].to_numpy()
            for lvl in levels
        ]
        if min(len(arr) for arr in groups) < 2:
            return {"testable": False, "notes": "kruskal_groups_too_small"}
        _, p = stats.kruskal(*groups)
        effect = eta_squared(groups)
        return {
            "testable": True,
            "n": int(len(pair2)),
            "observed_test": "kruskal",
            "effect": float(effect),
            "effect_name": "eta2_approx",
            "p_value": float(p),
            "notes": "",
        }

    if primary_test == "chi2":
        a = y.astype(str)
        b = x.astype(str)
        contingency = pd.crosstab(a, b)
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            return {"testable": False, "notes": "chi2_requires_at_least_2x2_table"}
        effect, p = cramers_v(contingency)
        return {
            "testable": True,
            "n": int(len(pair)),
            "observed_test": "chi2",
            "effect": float(effect),
            "effect_name": "cramers_v",
            "p_value": float(p),
            "notes": "",
        }

    return {
        "testable": False,
        "notes": f"unsupported_pairwise_test:{primary_test}",
    }


def run_regression_test(df: pd.DataFrame, validation_spec: dict, primary_test: str):
    outcome = validation_spec.get("outcome")
    predictors = validation_spec.get("predictors", [])
    controls = validation_spec.get("controls", [])

    if not outcome or not predictors:
        return {
            "testable": False,
            "notes": "validation_spec_missing_outcome_or_predictors",
        }

    built, joined, build_note = build_model_matrix(df, outcome, predictors, controls)
    if built is None:
        return {
            "testable": False,
            "notes": build_note,
        }

    y, X, y_type = built

    if len(joined) < 3:
        return {
            "testable": False,
            "notes": "too_few_complete_cases_for_regression",
        }

    non_constant_cols = [col for col in X.columns if X[col].nunique(dropna=True) > 1]
    X = X[non_constant_cols]

    if X.shape[1] == 0:
        return {
            "testable": False,
            "notes": "all_predictor_columns_constant",
        }

    X = X.apply(pd.to_numeric, errors="coerce").astype(float)
    y = pd.to_numeric(y, errors="coerce").astype(float)

    joined2 = pd.concat([y.rename("__y__"), X], axis=1).dropna()
    if len(joined2) < 3:
        return {
            "testable": False,
            "notes": "too_few_complete_cases_after_numeric_cleanup",
        }

    joined2 = joined2.apply(pd.to_numeric, errors="coerce").dropna()

    y = joined2["__y__"].astype(float)
    X = joined2.drop(columns=["__y__"]).astype(float)

    if X.shape[1] == 0:
        return {
            "testable": False,
            "notes": "empty_design_matrix_after_cleanup",
        }

    X = sm.add_constant(X, has_constant="add")
    X = X.astype(float)
    y = y.astype(float)

    predictor_cols = [
        c for c in X.columns
        if c != "const" and any(c == p or c.startswith(f"{p}_") for p in predictors)
    ]
    if not predictor_cols:
        return {
            "testable": False,
            "notes": "no_predictor_columns_in_design_matrix",
        }

    try:
        if primary_test == "linear_regression":
            if y_type != "numeric":
                return {
                    "testable": False,
                    "notes": "linear_regression_requires_numeric_outcome",
                }
            model = sm.OLS(y, X).fit()

        elif primary_test == "logistic_regression":
            if y_type != "binary":
                return {
                    "testable": False,
                    "notes": "logistic_regression_requires_binary_outcome",
                }
            if y.nunique() != 2:
                return {
                    "testable": False,
                    "notes": "binary_outcome_has_less_than_2_classes_after_cleanup",
                }
            model = sm.Logit(y, X).fit(disp=False)

        else:
            return {
                "testable": False,
                "notes": f"unsupported_regression_test:{primary_test}",
            }

    except Exception as e:
        return {
            "testable": False,
            "notes": f"regression_fit_failed:{type(e).__name__}:{str(e)}",
        }

    predictor_pvals = model.pvalues[predictor_cols]
    predictor_coefs = model.params[predictor_cols]

    min_p = float(predictor_pvals.min())
    main_col = predictor_pvals.idxmin()
    main_coef = float(predictor_coefs[main_col])

    return {
        "testable": True,
        "n": int(len(joined2)),
        "observed_test": primary_test,
        "effect": main_coef,
        "effect_name": "coef",
        "p_value": min_p,
        "main_predictor_column": main_col,
        "notes": "",
    }

def run_validation_test(df: pd.DataFrame, hypothesis: dict):
    validation_spec = hypothesis.get("validation_spec", {})
    primary_test = hypothesis.get("test_plan", {}).get("primary_test", "")

    if not isinstance(validation_spec, dict) or not validation_spec:
        return {
            "testable": False,
            "notes": "missing_validation_spec",
        }

    if primary_test in {"pearsonr", "spearmanr", "mannwhitneyu", "kruskal", "chi2"}:
        return run_pairwise_test(df, validation_spec, primary_test)

    if primary_test in {"linear_regression", "logistic_regression"}:
        return run_regression_test(df, validation_spec, primary_test)

    return {
        "testable": False,
        "notes": f"unsupported_test:{primary_test}",
    }


def run_bootstrap_for_hypothesis(df_test: pd.DataFrame, hypothesis: dict, args):
    rng = np.random.default_rng(args.bootstrap_random_state)
    effects = []
    threshold_passes = []
    direction_matches = []
    successful_runs = 0

    n_rows = len(df_test)
    if n_rows < args.min_n:
        return {
            "bootstrap_testable": False,
            "bootstrap_notes": "test_set_too_small_for_bootstrap",
        }

    for _ in range(args.bootstrap_iterations):
        sample_idx = rng.integers(0, n_rows, size=n_rows)
        df_boot = df_test.iloc[sample_idx].reset_index(drop=True)

        result = run_validation_test(df_boot, hypothesis)
        if not bool(result.get("testable", False)):
            continue

        effect = result.get("effect", np.nan)
        effect_name = result.get("effect_name", "")
        observed_test = result.get("observed_test", "")
        if pd.isna(effect):
            continue

        successful_runs += 1
        effects.append(float(effect))
        threshold_passes.append(effect_threshold_pass(effect_name, float(effect), args))
        direction_matches.append(
            safe_direction_match(
                hypothesis.get("direction", "unknown"),
                float(effect),
                observed_test,
            )
        )

    if successful_runs == 0:
        return {
            "bootstrap_testable": False,
            "bootstrap_notes": "no_successful_bootstrap_runs",
        }

    effects_arr = np.asarray(effects, dtype=float)
    ci_low = float(np.percentile(effects_arr, 2.5))
    ci_high = float(np.percentile(effects_arr, 97.5))
    median_effect = float(np.median(effects_arr))
    sign_consistency = float(max((effects_arr > 0).mean(), (effects_arr < 0).mean()))
    effect_pass_rate = float(np.mean(threshold_passes))
    direction_match_rate = float(np.mean(direction_matches))

    ci_excludes_zero = bool((ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0))
    bootstrap_stable = bool(
        ci_excludes_zero
        and sign_consistency >= args.bootstrap_sign_consistency_min
        and effect_pass_rate >= args.bootstrap_effect_pass_rate_min
    )

    return {
        "bootstrap_testable": True,
        "bootstrap_notes": "",
        "bootstrap_successful_runs": int(successful_runs),
        "bootstrap_median_effect": median_effect,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "bootstrap_ci_excludes_zero": ci_excludes_zero,
        "bootstrap_sign_consistency": sign_consistency,
        "bootstrap_effect_pass_rate": effect_pass_rate,
        "bootstrap_direction_match_rate": direction_match_rate,
        "bootstrap_stable": bootstrap_stable,
    }


def main():
    args = parse_args()

    hypotheses_path = Path(args.hypotheses_json)
    test_csv_path = Path(args.test_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = infer_dataset_name(hypotheses_path, args.dataset_name)

    hypotheses_obj = load_json(hypotheses_path)
    df_test = pd.read_csv(test_csv_path)

    raw_hypotheses = hypotheses_obj.get("hypotheses", [])
    cleaned_hypotheses, rejected_df = post_filter_hypotheses(raw_hypotheses, df_test)

    rows = []

    for h in cleaned_hypotheses:
        h_id = h.get("id", "")
        variables = h.get("variables", [])
        relationship_type = h.get("relationship_type", "")
        direction = h.get("direction", "unknown")
        claim = h.get("claim", "")
        supporting_evidence = h.get("supporting_evidence", [])
        test_plan = h.get("test_plan", {})
        primary_test = test_plan.get("primary_test", "")
        validation_spec = h.get("validation_spec", {})

        result = run_validation_test(df_test, h)

        row = {
            "id": h_id,
            "variables": json.dumps(variables, ensure_ascii=False),
            "relationship_type": relationship_type,
            "direction_expected": direction,
            "claim": claim,
            "supporting_evidence": json.dumps(supporting_evidence, ensure_ascii=False),
            "primary_test": primary_test,
            "validation_spec": json.dumps(validation_spec, ensure_ascii=False),
            "testable": bool(result.get("testable", False)),
            "n": result.get("n", np.nan),
            "observed_test": result.get("observed_test", ""),
            "effect": result.get("effect", np.nan),
            "effect_name": result.get("effect_name", ""),
            "p_value": result.get("p_value", np.nan),
            "main_predictor_column": result.get("main_predictor_column", ""),
            "direction_match": np.nan,
            "effect_threshold_pass": np.nan,
            "notes": result.get("notes", ""),
            "adjusted_p": np.nan,
            "bh_significant": False,
            "validated": False,
            "validation_status": "not_testable",
            "bootstrap_testable": np.nan,
            "bootstrap_successful_runs": np.nan,
            "bootstrap_median_effect": np.nan,
            "bootstrap_ci_low": np.nan,
            "bootstrap_ci_high": np.nan,
            "bootstrap_ci_excludes_zero": np.nan,
            "bootstrap_sign_consistency": np.nan,
            "bootstrap_effect_pass_rate": np.nan,
            "bootstrap_direction_match_rate": np.nan,
            "bootstrap_stable": np.nan,
            "final_accepted": False,
        }

        if row["testable"]:
            observed_effect = float(row["effect"])
            row["direction_match"] = safe_direction_match(direction, observed_effect, primary_test)
            row["effect_threshold_pass"] = effect_threshold_pass(
                row["effect_name"],
                observed_effect,
                args,
            )

        rows.append(row)

    results_df = pd.DataFrame(rows)

    valid_mask = (
        results_df["testable"].fillna(False)
        & results_df["p_value"].notna()
        & np.isfinite(pd.to_numeric(results_df["p_value"], errors="coerce"))
    )

    if valid_mask.any():
        pvals = results_df.loc[valid_mask, "p_value"].astype(float).to_numpy()
        qvals = benjamini_hochberg(pvals)
        results_df.loc[valid_mask, "adjusted_p"] = qvals
        results_df.loc[valid_mask, "bh_significant"] = results_df.loc[valid_mask, "adjusted_p"] < args.alpha

        for idx, row in results_df.loc[valid_mask].iterrows():
            if int(row["n"]) < args.min_n:
                status = "failed_min_n"
                validated = False
            elif not bool(row["bh_significant"]):
                status = "failed_bh"
                validated = False
            elif not bool(row["direction_match"]):
                status = "failed_direction"
                validated = False
            elif not bool(row["effect_threshold_pass"]):
                status = "failed_effect_threshold"
                validated = False
            else:
                status = "validated"
                validated = True

            results_df.at[idx, "validated"] = validated
            results_df.at[idx, "validation_status"] = status

    results_df.loc[~results_df["testable"].fillna(False), "validation_status"] = "not_testable"

    validated_ids = set(results_df.loc[results_df["validated"], "id"].tolist())

    for idx, row in results_df.iterrows():
        h_id = row["id"]
        if h_id not in validated_ids:
            continue

        hypothesis = next((h for h in cleaned_hypotheses if h.get("id") == h_id), None)
        if hypothesis is None:
            continue

        boot = run_bootstrap_for_hypothesis(df_test, hypothesis, args)

        results_df.at[idx, "bootstrap_testable"] = boot.get("bootstrap_testable", False)
        results_df.at[idx, "bootstrap_successful_runs"] = boot.get("bootstrap_successful_runs", np.nan)
        results_df.at[idx, "bootstrap_median_effect"] = boot.get("bootstrap_median_effect", np.nan)
        results_df.at[idx, "bootstrap_ci_low"] = boot.get("bootstrap_ci_low", np.nan)
        results_df.at[idx, "bootstrap_ci_high"] = boot.get("bootstrap_ci_high", np.nan)
        results_df.at[idx, "bootstrap_ci_excludes_zero"] = boot.get("bootstrap_ci_excludes_zero", np.nan)
        results_df.at[idx, "bootstrap_sign_consistency"] = boot.get("bootstrap_sign_consistency", np.nan)
        results_df.at[idx, "bootstrap_effect_pass_rate"] = boot.get("bootstrap_effect_pass_rate", np.nan)
        results_df.at[idx, "bootstrap_direction_match_rate"] = boot.get("bootstrap_direction_match_rate", np.nan)
        results_df.at[idx, "bootstrap_stable"] = boot.get("bootstrap_stable", False)

        if not bool(boot.get("bootstrap_testable", False)):
            results_df.at[idx, "validation_status"] = "failed_bootstrap_not_testable"
            results_df.at[idx, "final_accepted"] = False
            if not results_df.at[idx, "notes"]:
                results_df.at[idx, "notes"] = boot.get("bootstrap_notes", "")
        elif bool(boot.get("bootstrap_stable", False)):
            results_df.at[idx, "validation_status"] = "final_accepted"
            results_df.at[idx, "final_accepted"] = True
        else:
            results_df.at[idx, "validation_status"] = "failed_bootstrap_stability"
            results_df.at[idx, "final_accepted"] = False

    final_ids = set(results_df.loc[results_df["final_accepted"], "id"].tolist())
    final_hypotheses = [h for h in cleaned_hypotheses if h.get("id") in final_ids]
    validated_hypotheses = [h for h in cleaned_hypotheses if h.get("id") in validated_ids]

    final_obj = {"hypotheses": final_hypotheses}
    validated_obj = {"hypotheses": validated_hypotheses}
    cleaned_obj = {"hypotheses": cleaned_hypotheses}

    summary = {
        "dataset": dataset_name,
        "generated_hypotheses_raw": int(len(raw_hypotheses)),
        "hypotheses_after_pre_filter": int(len(cleaned_hypotheses)),
        "rejected_pre_validation": int(len(rejected_df)),
        "testable_hypotheses": int(results_df["testable"].fillna(False).sum()),
        "validated_hypotheses_before_bootstrap": int(results_df["validated"].sum()),
        "final_accepted_hypotheses": int(results_df["final_accepted"].sum()),
        "rejected_after_bootstrap": int(results_df["validated"].sum() - results_df["final_accepted"].sum()),
        "not_testable_hypotheses": int((~results_df["testable"].fillna(False)).sum()),
        "validation_rate_among_testable": (
            float(results_df["validated"].sum() / results_df["testable"].fillna(False).sum())
            if int(results_df["testable"].fillna(False).sum()) > 0 else 0.0
        ),
        "final_acceptance_rate_among_validated": (
            float(results_df["final_accepted"].sum() / results_df["validated"].sum())
            if int(results_df["validated"].sum()) > 0 else 0.0
        ),
        "alpha": args.alpha,
        "min_n": args.min_n,
        "min_abs_r": args.min_abs_r,
        "min_cramers_v": args.min_cramers_v,
        "min_eta2": args.min_eta2,
        "min_rank_biserial": args.min_rank_biserial,
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_sign_consistency_min": args.bootstrap_sign_consistency_min,
        "bootstrap_effect_pass_rate_min": args.bootstrap_effect_pass_rate_min,
    }

    results_path = output_dir / f"{dataset_name}_validation_results.csv"
    validated_path = output_dir / f"{dataset_name}_validated_hypotheses.json"
    final_path = output_dir / f"{dataset_name}_final_hypotheses.json"
    cleaned_path = output_dir / f"{dataset_name}_schema_cleaned_hypotheses.json"
    rejected_path = output_dir / f"{dataset_name}_pre_validation_rejections.csv"
    summary_path = output_dir / f"{dataset_name}_validation_summary.json"

    results_df.to_csv(results_path, index=False)
    rejected_df.to_csv(rejected_path, index=False)
    save_json(validated_obj, validated_path)
    save_json(final_obj, final_path)
    save_json(cleaned_obj, cleaned_path)
    save_json(summary, summary_path)

    print(f"Validation + bootstrap complete: {dataset_name}")
    print(f"Saved: {results_path}")
    print(f"Saved: {validated_path}")
    print(f"Saved: {final_path}")
    print(f"Saved: {cleaned_path}")
    print(f"Saved: {rejected_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()

#python validate_hypotheses.py --hypotheses-json outputs\hypotheses\adult_income_train_hypotheses.json --test-csv outputs\preprocess\adult_income_test_cleaned.csv --output-dir outputs\validation --min-abs-r 0.15 --min-cramers-v 0.20 --min-eta2 0.05 --min-rank-biserial 0.15 --bootstrap-iterations 200