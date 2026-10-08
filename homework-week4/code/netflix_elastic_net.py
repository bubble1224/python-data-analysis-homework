"""Week 4：Kaggle Netflix 数据的 Elastic Net 回归分析。"""

from pathlib import Path
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "netflix_titles.csv"
OUT = ROOT / "results"
URL = "https://raw.githubusercontent.com/MadoDoctor/EDA-Project-Netflix/main/netflix_titles.csv"


def load_data():
    DATA.parent.mkdir(exist_ok=True)
    if not DATA.exists():
        urllib.request.urlretrieve(URL, DATA)
    return pd.read_csv(DATA)


def prepare(df):
    out = pd.DataFrame(index=df.index)
    out["type"] = df["type"].fillna("Unknown")
    out["rating"] = df["rating"].fillna("Unknown")
    out["listed_in"] = df["listed_in"].fillna("Unknown").str.split(", ").str[0]
    country = df["country"].fillna("")
    genres = df["listed_in"].fillna("")
    out["country_count"] = country.str.count(",") + (country != "").astype(int)
    out["genre_count"] = genres.str.count(",") + (genres != "").astype(int)
    out["description_length"] = df["description"].fillna("").str.len()
    out["title_length"] = df["title"].fillna("").str.len()
    out["has_director"] = df["director"].notna().astype(int)
    out["has_cast"] = df["cast"].notna().astype(int)
    out["target"] = pd.to_numeric(df["release_year"], errors="coerce")
    return out.dropna(subset=["target"])


def main():
    df = prepare(load_data())
    X, y = df.drop(columns="target"), df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    model = ElasticNetCV(alphas=np.logspace(-3, 3, 100), l1_ratio=[.1, .3, .5, .7, .9, 1.0],
                         cv=KFold(10, shuffle=True, random_state=42), max_iter=100000, random_state=42)
    pipe = Pipeline([("preprocess", prep), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    selected = pd.DataFrame({"feature": names, "coefficient": model.coef_})
    selected["absolute_coefficient"] = selected["coefficient"].abs()
    selected = selected[selected["absolute_coefficient"] > 1e-8].sort_values("absolute_coefficient", ascending=False)
    OUT.mkdir(exist_ok=True)
    selected.to_csv(OUT / "netflix_elastic_net_selected_features.csv", index=False)
    metrics = {"samples": len(df), "alpha": model.alpha_, "l1_ratio": model.l1_ratio_,
               "test_RMSE": np.sqrt(mean_squared_error(y_test, pred)), "test_MAE": mean_absolute_error(y_test, pred),
               "test_R2": r2_score(y_test, pred), "nonzero_features": len(selected)}
    pd.DataFrame([metrics]).to_csv(OUT / "netflix_elastic_net_metrics.csv", index=False)
    top = selected.head(15).sort_values("coefficient")
    plt.figure(figsize=(10, 5)); plt.barh(top["feature"], top["coefficient"])
    plt.title("Netflix Elastic Net：绝对值最大的非零系数"); plt.xlabel("Standardized coefficient")
    plt.tight_layout(); plt.savefig(OUT / "netflix_elastic_net_coefficients.png", dpi=160)
    print(pd.Series(metrics).to_string())


if __name__ == "__main__":
    main()
