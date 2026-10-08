"""Week 2：Carseats 多元线性回归与 VIF。"""

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

URL = "https://vincentarelbundock.github.io/Rdatasets/csv/ISLR/Carseats.csv"


def main():
    df = pd.read_csv(URL).drop(columns=["Unnamed: 0"], errors="ignore")
    df["ShelveLoc"] = pd.Categorical(df["ShelveLoc"], categories=["Bad", "Medium", "Good"])
    formula = "Sales ~ Price + Income + Advertising + C(ShelveLoc, Treatment(reference='Bad'))"
    model = smf.ols(formula, data=df).fit()
    print(model.summary())
    name = "C(ShelveLoc, Treatment(reference='Bad'))[T.Good]"
    print(f"\nShelveLoc 基准组：Bad")
    print(f"ShelveLoc[Good] 系数：{model.params[name]:.4f}")
    print("含义：控制其他变量后，Good 货架位置的预测 Sales 比 Bad 平均高该系数（千件）。")
    exog = pd.DataFrame(model.model.exog, columns=model.model.exog_names)
    rows = [{"Variable": col, "VIF": variance_inflation_factor(exog.values, i)}
            for i, col in enumerate(exog.columns) if col != "Intercept"]
    vif = pd.DataFrame(rows)
    print("\nVIF：")
    print(vif.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("结论：最大 VIF < 5 时，没有明显多重共线性风险。")


if __name__ == "__main__":
    main()
