import os
import pandas as pd
from sklearn.datasets import fetch_california_housing

BASE_PATH = r"D:\THESIS\DATA"
os.makedirs(BASE_PATH, exist_ok=True)

def save(df, name):
    df.to_csv(os.path.join(BASE_PATH, name), index=False)
    print(f"Saved {name} | shape={df.shape}")

# ---------- 1. Adult Income ----------
adult_cols = [
    "age","workclass","fnlwgt","education","education_num",
    "marital_status","occupation","relationship","race","sex",
    "capital_gain","capital_loss","hours_per_week","native_country","income"
]

adult = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    names=adult_cols,
    sep=", ",
    engine="python",
    na_values="?"
)

save(adult, "adult_income.csv")

# ---------- 2. Wine Quality ----------
wine_red = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    sep=";"
)
wine_white = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
    sep=";"
)

save(wine_red, "wine_quality_red.csv")
save(wine_white, "wine_quality_white.csv")

# ---------- 3. Credit Card Default ----------
try:
    credit = pd.read_excel(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls",
        header=1,
        engine="xlrd"
    )
    save(credit, "credit_card_default.csv")
except Exception as e:
    print("FAILED: Credit Card Default dataset")
    print(e)

# ---------- 4. California Housing ----------
try:
    california = fetch_california_housing(as_frame=True)
    save(california.frame, "california_housing.csv")
except Exception as e:
    print("FAILED: California Housing dataset")
    print(e)

print("Done.")
