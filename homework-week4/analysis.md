# Week 4：Netflix 数据 Elastic Net 分析

## 研究问题

使用 Kaggle 的 **Netflix Movies and TV Shows** 数据，分析节目元数据能否预测内容的 `release_year`，并观察 Elastic Net 如何在类别变量和文本结构变量中进行收缩与筛选。

## 数据与方法

- 删除 `release_year` 缺失记录；
- 使用 `type`、`rating`、`listed_in` 的首个类别作为分类变量；
- 将国家数、类型数、标题长度、简介长度，以及是否有导演/演员信息转成数值变量；
- 训练集/测试集按 80%/20% 划分，随机种子为 42；
- 在 Pipeline 中完成缺失值填补、独热编码和 Z-score 标准化，避免数据泄漏；
- 使用 `ElasticNetCV`，10 折交叉验证，同时搜索 `alpha` 和 `l1_ratio`。

运行：

```bash
python homework-week4/code/netflix_elastic_net.py
```

结果写入 `homework-week4/results/`：

- `netflix_elastic_net_metrics.csv`：样本量、最优超参数和测试集指标；
- `netflix_elastic_net_selected_features.csv`：非零系数及其绝对值；
- `netflix_elastic_net_coefficients.png`：绝对系数最大的特征图。

一次可复现运行的结果约为：样本量 6234，最优 `alpha=0.002656`，`l1_ratio=0.90`，测试集 RMSE 为 6.753，MAE 为 4.181，R² 为 0.286，保留 48 个非零特征。

## 结论

最优 `l1_ratio` 接近 1，说明当前特征设计下 L1 稀疏作用较强，但 Elastic Net 仍通过 L2 项提升了相关特征下的稳定性。模型将大量 one-hot 特征压缩为 0，保留的特征主要与节目类型、分级和内容结构有关。测试集 R² 约为 0.286，说明这些元数据能够解释部分发行年份差异，但不能替代完整的时间、内容和业务信息。系数表示控制其他变量后的条件关联，不应直接解释为因果效应。
