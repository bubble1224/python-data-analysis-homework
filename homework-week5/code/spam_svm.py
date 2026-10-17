"""Week 5：Spambase 垃圾邮件数据的 SVM 分类（课件案例①）。"""

from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
RANDOM_STATE = 42


def load_data():
    # Spambase：4601 封邮件、57 个特征（词频 + 大写字母比例 + 符号频率），1 表示垃圾邮件
    data = fetch_openml(data_id=44, as_frame=True)
    X = data.data.astype(float)
    y = data.target.astype(int)
    return X, y


def search(X_train, y_train, kernel, grid):
    # 标准化放进 Pipeline，交叉验证每一折只用该折训练数据拟合，避免数据泄漏
    pipe = Pipeline([("scale", StandardScaler()), ("model", SVC(kernel=kernel))])
    grid_search = GridSearchCV(pipe, grid, cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
                               scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=RANDOM_STATE)

    linear_grid = {"model__C": [.01, .1, 1, 10, 100]}
    rbf_grid = {"model__C": [.1, 1, 10, 100], "model__gamma": ["scale", .001, .01, .1]}
    rows, curves = [], {}

    for name, kernel, grid in [("linear SVM", "linear", linear_grid), ("RBF SVM", "rbf", rbf_grid)]:
        best = search(X_train, y_train, kernel, grid)
        row = {"model": name, "kernel": kernel, "best_C": best.best_params_["model__C"],
               "best_gamma": best.best_params_.get("model__gamma", "-"),
               "cv_accuracy": best.best_score_}
        start = time.perf_counter()
        best.best_estimator_.fit(X_train, y_train)
        row["train_seconds"] = time.perf_counter() - start
        decision = best.best_estimator_.decision_function(X_test)
        row["test_accuracy"] = accuracy_score(y_test, best.best_estimator_.predict(X_test))
        row["test_auc"] = roc_auc_score(y_test, decision)
        row["support_vectors"] = int(best.best_estimator_.named_steps["model"].n_support_.sum())
        rows.append(row)
        curves[name] = roc_curve(y_test, decision)

    baseline = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=5000))])
    baseline.fit(X_train, y_train)
    decision = baseline.decision_function(X_test)
    rows.append({"model": "logistic regression", "kernel": "-", "best_C": "-", "best_gamma": "-",
                 "cv_accuracy": np.nan, "train_seconds": np.nan,
                 "test_accuracy": accuracy_score(y_test, baseline.predict(X_test)),
                 "test_auc": roc_auc_score(y_test, decision), "support_vectors": np.nan})
    curves["logistic regression"] = roc_curve(y_test, decision)

    OUT.mkdir(exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "spam_svm_metrics.csv", index=False)

    plt.figure(figsize=(7, 5))
    for name, (fpr, tpr, _) in curves.items():
        auc = frame.loc[frame["model"] == name, "test_auc"].iloc[0]
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=.5)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("Spambase: SVM kernels vs logistic regression")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT / "spam_svm_roc.png", dpi=160)

    support_share = frame.loc[frame["model"] == "RBF SVM", "support_vectors"].iloc[0] / len(X_train)
    print(frame.to_string(index=False))
    print(f"\nRBF SVM 支持向量占训练集比例：{support_share:.1%}，其余样本删掉后边界不变（KKT 互补松弛带来的稀疏性）。")


if __name__ == "__main__":
    main()
