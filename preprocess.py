import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


class DeterministicPreprocessor:
    def __init__(
        self,
        high_cardinality_threshold=20,
        missing_threshold=0.4,
        add_outlier_flags=True,
        add_missing_indicators=True,
        min_stats_n=30,
        id_like_unique_ratio=0.90,
        id_like_max_dup_ratio=0.05,
        skew_mean_threshold=0.5,
        outlier_rate_mean_threshold=0.05,
        drop_id_like=True,
        drop_high_missing=True,
        drop_constant=True,
        categorical_missing_token="missing",
        treat_small_numeric_as_ordinal=True,
        small_numeric_unique_max=10,
    ):
        self.high_cardinality_threshold = int(high_cardinality_threshold)
        self.missing_threshold = float(missing_threshold)
        self.add_outlier_flags = bool(add_outlier_flags)
        self.add_missing_indicators = bool(add_missing_indicators)
        self.min_stats_n = int(min_stats_n)

        self.id_like_unique_ratio = float(id_like_unique_ratio)
        self.id_like_max_dup_ratio = float(id_like_max_dup_ratio)

        self.skew_mean_threshold = float(skew_mean_threshold)
        self.outlier_rate_mean_threshold = float(outlier_rate_mean_threshold)

        self.drop_id_like = bool(drop_id_like)
        self.drop_high_missing = bool(drop_high_missing)
        self.drop_constant = bool(drop_constant)

        self.categorical_missing_token = str(categorical_missing_token)

        self.treat_small_numeric_as_ordinal = bool(treat_small_numeric_as_ordinal)
        self.small_numeric_unique_max = int(small_numeric_unique_max)

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.columns = (
            out.columns.astype(str)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace(r"[^a-z0-9_]", "", regex=True)
        )
        return out

    def _is_integer_like_series(self, s: pd.Series) -> bool:
        if not pd.api.types.is_numeric_dtype(s):
            return False
        x = s.dropna()
        if x.empty:
            return False
        if pd.api.types.is_integer_dtype(x):
            return True
        frac = (x.astype(float) - np.floor(x.astype(float))).abs()
        return bool((frac < 1e-9).mean() > 0.99)

    def _detect_id_like(self, s: pd.Series) -> bool:
        x = s.dropna()
        n = int(len(x))
        if n == 0:
            return False

        nunique = int(x.nunique(dropna=True))
        unique_ratio = nunique / n

        if pd.api.types.is_numeric_dtype(x) and self._is_integer_like_series(x):
            if unique_ratio >= self.id_like_unique_ratio:
                return True

        if pd.api.types.is_object_dtype(x) or pd.api.types.is_string_dtype(x):
            if unique_ratio >= self.id_like_unique_ratio:
                dup_ratio = 1.0 - unique_ratio
                if dup_ratio <= self.id_like_max_dup_ratio:
                    return True

        return False

    def _infer_feature_type(self, s: pd.Series) -> str:
        x = s.dropna()
        nunique = int(x.nunique(dropna=True))

        if nunique <= 1:
            return "constant"

        if self._detect_id_like(s):
            return "id_like"

        if s.dtype == "bool":
            return "binary"

        if pd.api.types.is_numeric_dtype(s):
            if nunique == 2:
                return "binary"
            if self.treat_small_numeric_as_ordinal and nunique <= self.small_numeric_unique_max:
                return "numeric_ordinal"
            return "numeric_continuous"

        if nunique == 2:
            return "binary"
        if nunique > self.high_cardinality_threshold:
            return "categorical_high_cardinality"
        return "categorical_low_cardinality"

    def _compute_outlier_info_iqr(self, x_non_null: pd.Series) -> dict:
        x = x_non_null.dropna()
        n = int(len(x))

        if n < 5:
            return {
                "method": "iqr",
                "n_used": n,
                "confidence": "low_n",
                "lower_bound": None,
                "upper_bound": None,
                "outlier_rate": None,
            }

        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1

        if pd.isna(iqr) or float(iqr) == 0.0:
            return {
                "method": "iqr",
                "n_used": n,
                "confidence": "iqr_zero",
                "lower_bound": None,
                "upper_bound": None,
                "outlier_rate": 0.0,
            }

        lower = float(q1 - 1.5 * iqr)
        upper = float(q3 + 1.5 * iqr)
        mask = (x < lower) | (x > upper)

        return {
            "method": "iqr",
            "n_used": n,
            "confidence": "ok" if n >= self.min_stats_n else "medium_n",
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_rate": float(mask.mean()),
        }

    def profile_df(self, df: pd.DataFrame) -> dict:
        df_norm = self._normalize_columns(df)

        profile = {}
        for col in df_norm.columns:
            s = df_norm[col]
            missing_rate = float(s.isna().mean())
            dtype_str = str(s.dtype)

            x = s.dropna()
            n_total = int(len(s))
            n_non_null = int(len(x))
            nunique = int(x.nunique(dropna=True)) if n_non_null > 0 else 0

            feature_type = self._infer_feature_type(s)

            item = {
                "feature_type": feature_type,
                "dtype": dtype_str,
                "n_total": n_total,
                "n_non_null": n_non_null,
                "missing_rate": round(missing_rate, 6),
                "n_unique": nunique,
            }

            if pd.api.types.is_numeric_dtype(s) and feature_type not in {"id_like", "constant"}:
                skew = float(x.skew()) if n_non_null >= 3 else None
                out_info = self._compute_outlier_info_iqr(x)

                item.update(
                    {
                        "skewness": None if skew is None else round(skew, 6),
                        "outlier_method": out_info["method"],
                        "outlier_n_used": out_info["n_used"],
                        "outlier_confidence": out_info["confidence"],
                        "outlier_rate": None if out_info["outlier_rate"] is None else round(float(out_info["outlier_rate"]), 6),
                        "lower_bound": out_info["lower_bound"],
                        "upper_bound": out_info["upper_bound"],
                    }
                )

            issues = []
            suggested = []

            if feature_type == "constant":
                issues.append("constant")
                if self.drop_constant:
                    suggested.append("drop_constant")

            if feature_type == "id_like":
                issues.append("id_like")
                if self.drop_id_like:
                    suggested.append("drop_id_like")

            if missing_rate > self.missing_threshold:
                issues.append("high_missing")
                if self.drop_high_missing:
                    suggested.append("drop_high_missing")

            item["issues"] = issues
            item["suggested_actions"] = suggested
            item["preprocessing_leakage_note"] = (
                "Profiling is read-only. Train preprocessing state must be reused when transforming test data."
            )
            profile[col] = item

        return profile

    def fit_preprocessing_state(self, df: pd.DataFrame, profile: dict) -> dict:
        df_norm = self._normalize_columns(df)

        state = {
            "policy": {
                "missing_threshold": self.missing_threshold,
                "drop_constant": self.drop_constant,
                "drop_id_like": self.drop_id_like,
                "drop_high_missing": self.drop_high_missing,
                "add_outlier_flags": self.add_outlier_flags,
                "add_missing_indicators": self.add_missing_indicators,
                "skew_mean_threshold": self.skew_mean_threshold,
                "outlier_rate_mean_threshold": self.outlier_rate_mean_threshold,
                "min_stats_n": self.min_stats_n,
                "categorical_missing_token": self.categorical_missing_token,
            },
            "columns": {},
            "raw_input_columns": list(df_norm.columns),
        }

        for col in df_norm.columns:
            prof = profile[col]
            ft = prof.get("feature_type")
            mr = float(prof.get("missing_rate", 0.0))
            s = df_norm[col]

            drop_actions = []
            if ft == "constant" and self.drop_constant:
                drop_actions.append("dropped_constant")
            if ft == "id_like" and self.drop_id_like:
                drop_actions.append("dropped_id_like")
            if mr > self.missing_threshold and self.drop_high_missing:
                drop_actions.append("dropped_high_missing")

            col_state = {
                "feature_type": ft,
                "dtype": prof.get("dtype"),
                "missing_rate": mr,
                "drop": bool(drop_actions),
                "drop_actions": drop_actions,
                "add_missing_indicator": bool(self.add_missing_indicators and mr > 0.0),
            }

            if not col_state["drop"]:
                if pd.api.types.is_numeric_dtype(s) and ft not in {"id_like", "constant"}:
                    x = s.dropna()
                    skew = prof.get("skewness")
                    out_rate = prof.get("outlier_rate")

                    use_mean = False
                    if skew is not None and out_rate is not None:
                        try:
                            use_mean = (
                                abs(float(skew)) < self.skew_mean_threshold
                                and float(out_rate) < self.outlier_rate_mean_threshold
                            )
                        except Exception:
                            use_mean = False

                    if use_mean:
                        fill_value = float(x.mean()) if len(x) > 0 else 0.0
                        impute_method = "mean"
                    else:
                        fill_value = float(x.median()) if len(x) > 0 else 0.0
                        impute_method = "median"

                    col_state["imputation"] = {
                        "method": impute_method,
                        "value": fill_value,
                    }

                    if self.add_outlier_flags:
                        col_state["outlier_flag"] = {
                            "enabled": True,
                            "method": "iqr",
                            "lower_bound": prof.get("lower_bound"),
                            "upper_bound": prof.get("upper_bound"),
                        }
                else:
                    col_state["imputation"] = {
                        "method": "constant_missing",
                        "value": self.categorical_missing_token,
                    }

            state["columns"][col] = col_state

        output_columns = []
        for col in df_norm.columns:
            col_state = state["columns"][col]
            if col_state["drop"]:
                continue

            output_columns.append(col)
            if col_state.get("add_missing_indicator", False):
                output_columns.append(f"{col}__is_missing")

            outlier_flag = col_state.get("outlier_flag", {})
            if outlier_flag.get("enabled", False):
                output_columns.append(f"{col}__is_outlier")

        state["output_columns"] = output_columns
        return state

    def transform_with_state(self, df: pd.DataFrame, state: dict) -> tuple[pd.DataFrame, dict]:
        df_norm = self._normalize_columns(df)
        raw_expected = state.get("raw_input_columns", [])

        working = pd.DataFrame(index=df_norm.index)
        for col in raw_expected:
            if col in df_norm.columns:
                working[col] = df_norm[col]
            else:
                working[col] = pd.Series([pd.NA] * len(df_norm), index=df_norm.index)

        log = {
            "dropped": {},
            "imputed": {},
            "engineered": {},
            "notes": [],
        }

        cleaned = pd.DataFrame(index=working.index)

        for col in raw_expected:
            col_state = state["columns"][col]

            if col_state["drop"]:
                log["dropped"][col] = {
                    "actions": col_state["drop_actions"],
                    "feature_type": col_state["feature_type"],
                    "missing_rate_train": col_state["missing_rate"],
                }
                continue

            s = working[col]

            if col_state.get("add_missing_indicator", False):
                miss_col = f"{col}__is_missing"
                cleaned[miss_col] = s.isna().astype(int)
                log["engineered"][miss_col] = {
                    "source": col,
                    "type": "missing_indicator",
                }

            impute = col_state.get("imputation", {})
            method = impute.get("method")
            value = impute.get("value")

            if method in {"mean", "median"}:
                cleaned[col] = pd.to_numeric(s, errors="coerce").fillna(value)
                log["imputed"][col] = {"method": method, "value": value}
            elif method == "constant_missing":
                cleaned[col] = s.fillna(value)
                log["imputed"][col] = {"method": method, "value": value}
            else:
                cleaned[col] = s
                log["notes"].append(f"no_imputation_rule:{col}")

            outlier_flag = col_state.get("outlier_flag", {})
            if outlier_flag.get("enabled", False):
                flag_col = f"{col}__is_outlier"
                lb = outlier_flag.get("lower_bound")
                ub = outlier_flag.get("upper_bound")

                if lb is None or ub is None:
                    cleaned[flag_col] = 0
                    log["engineered"][flag_col] = {
                        "source": col,
                        "type": "outlier_flag",
                        "status": "skipped_no_bounds",
                    }
                else:
                    numeric_col = pd.to_numeric(cleaned[col], errors="coerce")
                    mask = (numeric_col < float(lb)) | (numeric_col > float(ub))
                    cleaned[flag_col] = mask.astype(int)
                    log["engineered"][flag_col] = {
                        "source": col,
                        "type": "outlier_flag",
                        "method": "iqr",
                        "lower_bound": float(lb),
                        "upper_bound": float(ub),
                        "outlier_rate_on_transformed": float(mask.mean()),
                    }

        output_columns = state.get("output_columns", list(cleaned.columns))
        cleaned = cleaned.reindex(columns=output_columns)

        return cleaned, log

    def preprocess_train(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict, dict]:
        profile = self.profile_df(df)
        state = self.fit_preprocessing_state(df, profile)
        cleaned, log = self.transform_with_state(df, state)
        return cleaned, profile, log, state


def save_json(obj: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw input CSV")
    parser.add_argument("--output-dir", required=True, help="Directory to save preprocessing outputs")
    parser.add_argument(
        "--mode",
        choices=["fit_transform", "transform_only"],
        default="fit_transform",
        help="fit_transform for train, transform_only for test",
    )
    parser.add_argument(
        "--state-json",
        default=None,
        help="Required for transform_only. Path to train preprocessing_state.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = input_path.stem
    df = pd.read_csv(input_path)

    preprocessor = DeterministicPreprocessor()

    if args.mode == "fit_transform":
        clean_df, profile, log, state = preprocessor.preprocess_train(df)

        clean_path = output_dir / f"{base_name}_cleaned.csv"
        profile_path = output_dir / f"{base_name}_profile.json"
        log_path = output_dir / f"{base_name}_transform_log.json"
        state_path = output_dir / f"{base_name}_preprocessing_state.json"
        summary_path = output_dir / f"{base_name}_preprocess_summary.json"

        clean_df.to_csv(clean_path, index=False)
        save_json(profile, profile_path)
        save_json(log, log_path)
        save_json(state, state_path)

        summary = {
            "dataset": base_name,
            "mode": args.mode,
            "input_rows": int(df.shape[0]),
            "input_cols": int(df.shape[1]),
            "output_rows": int(clean_df.shape[0]),
            "output_cols": int(clean_df.shape[1]),
            "dropped_columns": list(log["dropped"].keys()),
            "engineered_columns": list(log["engineered"].keys()),
        }
        save_json(summary, summary_path)

        print(f"Train preprocessing complete: {base_name}")
        print(f"Saved: {clean_path}")
        print(f"Saved: {profile_path}")
        print(f"Saved: {log_path}")
        print(f"Saved: {state_path}")
        print(f"Saved: {summary_path}")

    else:
        if not args.state_json:
            raise ValueError("--state-json is required for transform_only mode")

        state = load_json(Path(args.state_json))
        clean_df, log = preprocessor.transform_with_state(df, state)

        clean_path = output_dir / f"{base_name}_cleaned.csv"
        log_path = output_dir / f"{base_name}_transform_log.json"
        summary_path = output_dir / f"{base_name}_preprocess_summary.json"

        clean_df.to_csv(clean_path, index=False)
        save_json(log, log_path)

        summary = {
            "dataset": base_name,
            "mode": args.mode,
            "input_rows": int(df.shape[0]),
            "input_cols": int(df.shape[1]),
            "output_rows": int(clean_df.shape[0]),
            "output_cols": int(clean_df.shape[1]),
            "dropped_columns": list(log["dropped"].keys()),
            "engineered_columns": list(log["engineered"].keys()),
        }
        save_json(summary, summary_path)

        print(f"Apply preprocessing complete: {base_name}")
        print(f"Saved: {clean_path}")
        print(f"Saved: {log_path}")
        print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()

#python preprocess.py --input outputs\data\adult_income_train.csv --output-dir outputs\preprocess --mode fit_transform
#python preprocess.py --input outputs\data\adult_income_test.csv --output-dir outputs\preprocess --mode transform_only --state-json outputs\preprocess\adult_income_train_preprocessing_state.json