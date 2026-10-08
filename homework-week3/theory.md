# Week 3：正则化回归作业 1、3、4

本周根据《正则化回归交互式课件_Ridge_Lasso.html》完成最后的作业 1、3、4。

## 作业 1：正交设计下 Ridge 与 OLS 的关系

不含截距，且 `XᵀX = I` 时，OLS 闭合解为

\[
\hat{\beta}_{OLS}=(X^TX)^{-1}X^Ty=X^Ty.
\]

Ridge 的闭合解为

\[
\hat{\beta}_{Ridge}=(X^TX+\lambda I)^{-1}X^Ty.
\]

代入正交条件：

\[
\begin{aligned}
\hat{\beta}_{Ridge}
&=((1+\lambda)I)^{-1}X^Ty\\
&=\frac{1}{1+\lambda}X^Ty\\
&=\boxed{\frac{1}{1+\lambda}\hat{\beta}_{OLS}}.
\end{aligned}
\]

所以 Ridge 在正交设计下对所有 OLS 系数进行相同倍数的收缩；`λ=0` 时恢复 OLS，有限的 `λ>0` 通常不会把系数精确变成 0。

## 作业 3：为什么正则化前必须 Z-score 标准化

对每个特征执行 `z_j = (x_j - mean(x_j)) / sd(x_j)`，主要原因如下：

1. Ridge 的 `L2` 惩罚和 Lasso 的 `L1` 惩罚直接作用于系数。如果特征量纲不同，系数大小会同时反映单位，而不只是反映变量贡献，惩罚就不公平。
2. 同一变量换单位会改变系数数值。比如把收入从“元”换成“千元”，系数约缩小 1000 倍；未标准化时，正则化结果会随单位变化。
3. 未中心化时截距可能吸收均值，实际建模应让标准化和正则化处于同一 Pipeline 中，并且只用训练集拟合标准化参数，避免数据泄漏。
4. 标准化还能改善数值优化的条件数，使不同方向的惩罚尺度更可比较。

不标准化的后果是：某些变量因为单位大或小而被过度收缩或保留，变量筛选结果可能被量纲“绑架”，系数路径和模型解释也失去可比性。

## 作业 4：Hitters 上比较 Ridge、Lasso、Elastic Net

`code/regularized_regression_comparison.py` 使用公开 ISLR Hitters 数据：删除 `Salary` 缺失记录，对分类变量独热编码，在训练集上标准化，并使用 10 折交叉验证分别训练 `RidgeCV`、`LassoCV` 和 `ElasticNetCV`。

程序输出：

- 测试集 RMSE；
- 最优 `alpha`，以及 Elastic Net 的 `l1_ratio`；
- 非零系数数量；
- 三种模型的系数路径图；
- Lasso 与 Elastic Net 的 1-SE 候选 `alpha`。

1-SE 法则先找到最小 CV 误差，再选择误差不超过 `MSE_min + SE_min` 的候选中正则化更强的模型。这样通常能用很小的预测性能代价换取更少的变量和更稳定的解释。Ridge 通常仍保留所有非零系数，因此它的“简化”主要体现为更强收缩。
