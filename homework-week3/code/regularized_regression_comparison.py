"""Week 3：在 Hitters 数据上比较 Ridge、Lasso 与 Elastic Net。"""

from pathlib import Path
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

DATA_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/ISLR/Hitters.csv"
OUT = Path(__file__).resolve().parent.parent / "results"


def one_se_alpha(model):
    path = np.asarray(model.mse_path_)
    if path.ndim == 3:
        path = path.min(axis=2)
    mean = path.mean(axis=1)
    se = path.std(axis=1, ddof=1) / np.sqrt(path.shape[1])
    best = int(np.argmin(mean))
    eligible = np.flatnonzero(mean <= mean[best] + se[best])
    idx = eligible[np.argmax(model.alphas_[eligible])]
    return float(model.alphas_[idx]), float(mean[best]), float(se[best])


def main():
    df = pd.read_csv(DATA_URL).drop(columns=["Unnamed: 0"], errors="ignore").dropna(subset=["Salary"])
    X, y = df.drop(columns="Salary"), df["Salary"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", make_pipeline(OneHotEncoder(handle_unknown="ignore", sparse_output=False), StandardScaler()), categorical),
    ])
    alphas = np.logspace(-3, 4, 120)
    cv = KFold(10, shuffle=True, random_state=42)
    models = {
        "Ridge": RidgeCV(alphas=alphas, cv=cv),
        "Lasso": LassoCV(alphas=alphas, cv=cv, max_iter=100000, random_state=42),
        "Elastic Net": ElasticNetCV(alphas=alphas, l1_ratio=[.2, .5, .8], cv=cv, max_iter=100000, random_state=42),
    }
    fitted, rows = {}, []
    for name, estimator in models.items():
        pipe = make_pipeline(preprocess, estimator)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        fitted[name] = pipe
        rows.append({"Model": name, "Test RMSE": np.sqrt(mean_squared_error(y_test, pred)),
                     "Non-zero variables": int(np.count_nonzero(np.abs(estimator.coef_) > 1e-8)),
                     "Alpha": estimator.alpha_, "L1 ratio": getattr(estimator, "l1_ratio_", np.nan)})
    OUT.mkdir(exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "regularized_regression_metrics.csv", index=False)
    print(result.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    for name in ["Lasso", "Elastic Net"]:
        est = fitted[name].named_steps[name.lower().replace(" ", "")] if False else fitted[name].steps[-1][1]
        alpha, mse, se = one_se_alpha(est)
        print(f"{name}: CV 最小 MSE={mse:.2f}, SE={se:.2f}, 1-SE alpha={alpha:.5f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    Xtr = fitted["Ridge"].steps[0][1].transform(X_train)
    for ax, (name, pipe) in zip(axes, fitted.items()):
        est = pipe.steps[-1][1]
        # RidgeCV 在部分 scikit-learn 版本中不暴露 alphas_，使用统一网格。
        grid = np.asarray(est.alphas_ if hasattr(est, "alphas_") else alphas)
        values = []
        for alpha in grid:
            if name == "Ridge": model = RidgeCV(alphas=[alpha]).fit(Xtr, y_train)
            elif name == "Lasso": model = LassoCV(alphas=[alpha], cv=3, max_iter=100000).fit(Xtr, y_train)
            else: model = ElasticNetCV(alphas=[alpha], l1_ratio=[est.l1_ratio_], cv=3, max_iter=100000).fit(Xtr, y_train)
            values.append(model.coef_)
        ax.plot(np.log10(grid), np.asarray(values), alpha=.55, linewidth=.7)
        ax.axvline(np.log10(est.alpha_), color="black", linestyle="--")
        ax.set_title(name); ax.set_xlabel("log10(alpha)"); ax.grid(alpha=.25)
    axes[0].set_ylabel("Coefficient")
    fig.suptitle("Hitters：Ridge / Lasso / Elastic Net 系数路径")
    fig.tight_layout(); fig.savefig(OUT / "regularized_coefficient_paths.png", dpi=160)


if __name__ == "__main__":
    main()
