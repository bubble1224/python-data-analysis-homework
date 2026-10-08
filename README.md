# Python 数据分析课程作业

本仓库整理《Python 数据分析》课程 Week 1–Week 4 作业，包含理论推导、可复现 Python 代码、结果文件和分析说明。

## 作业目录

| 周次 | 内容 | 入口 |
|---|---|---|
| Week 1 | Python 基础输出 | [homework-week1](homework-week1) |
| Week 2 | OLS 残差性质、调整后 R²、Carseats 多元回归与 VIF | [homework-week2](homework-week2) |
| Week 3 | 正则化回归作业 1、3、4：Ridge、Lasso、Elastic Net | [homework-week3](homework-week3) |
| Week 4 | Kaggle Netflix 数据 Elastic Net 分析 | [homework-week4](homework-week4) |

## 环境安装

建议使用 Python 3.10 或更高版本：

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Week 2 的 `Carseats` 数据、Week 3 的 `Hitters` 数据和 Week 4 的 Netflix 数据会从公开数据源下载；首次运行需要网络连接。

## 运行方式

```bash
python homework-week1/hello.py
python homework-week2/code/carseats_regression.py
python homework-week3/code/regularized_regression_comparison.py
python homework-week4/code/netflix_elastic_net.py
```

各实验的图表和指标会写入对应目录下的 `results/`。数据下载缓存位于 Week 4 的 `data/`，已通过 `.gitignore` 排除，避免把外部原始数据重复提交到仓库。

## Week 3 作业说明

- **作业 1**：在正交设计 `XᵀX=I` 下，将 Ridge 闭合解化简为 `β̂_Ridge = β̂_OLS/(1+λ)`。
- **作业 3**：说明正则化前 Z-score 标准化的必要性、量纲影响、截距处理和数据泄漏问题。
- **作业 4**：使用 Hitters 数据，以 10 折交叉验证比较 `RidgeCV`、`LassoCV` 和 `ElasticNetCV`，输出测试集 RMSE、非零系数数目、系数路径图，并用 1-SE 法则讨论更稀疏的选择。

详细解答见 [homework-week3/theory.md](homework-week3/theory.md)。

## Week 4 作业说明

使用 Kaggle 的 **Netflix Movies and TV Shows** 数据集，仅保留电影记录，以电影时长（分钟）为响应变量，使用发行年份、上架日期、分级、国家和类型等字段作为解释变量。预处理、标准化、独热编码和 `ElasticNetCV` 放在同一 Pipeline 中，使用 10 折交叉验证选择 `alpha` 与 `l1_ratio`。

详细分析见 [homework-week4/analysis.md](homework-week4/analysis.md)。

## 目录结构

```text
.
├── README.md
├── requirements.txt
├── homework-week1/hello.py
├── homework-week2/
│   ├── theory.md
│   └── code/carseats_regression.py
├── homework-week3/
│   ├── theory.md
│   ├── code/regularized_regression_comparison.py
│   └── results/
└── homework-week4/
    ├── analysis.md
    ├── code/netflix_elastic_net.py
    └── results/
```
