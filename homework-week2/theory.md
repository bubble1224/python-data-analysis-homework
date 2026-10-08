# Week 2：线性回归基础

## 1. OLS 残差的两个正交性质

带截距的一元线性回归为 `y_i = β₀ + β₁x_i + e_i`，残差平方和为

\[
RSS=\sum_i(y_i-\beta_0-\beta_1x_i)^2.
\]

对 `β₀` 求偏导并令其为零，得到

\[
-2\sum_i(y_i-\hat\beta_0-\hat\beta_1x_i)=0
\quad\Rightarrow\quad \sum_i e_i=0.
\]

对 `β₁` 求偏导并令其为零，得到

\[
-2\sum_i x_i(y_i-\hat\beta_0-\hat\beta_1x_i)=0
\quad\Rightarrow\quad \sum_i x_i e_i=0.
\]

因此，在包含截距的 OLS 回归中，残差和为零，且残差与解释变量正交。

## 2. 普通 R² 与调整后 R²

普通决定系数为 `R² = 1 - RSS/TSS`。加入变量后，旧模型可以通过令新变量系数为 0 得到，所以新的 RSS 不会增加，而 TSS 不变，普通 R² 只能上升或保持不变。

调整后决定系数为

\[
R^2_{adj}=1-\frac{RSS/(n-p-1)}{TSS/(n-1)}.
\]

新增变量会减少一个残差自由度。如果 RSS 的下降不足以抵消自由度减少带来的惩罚，调整后 R² 就会下降。因此调整后 R² 更适合比较变量数量不同的模型。
