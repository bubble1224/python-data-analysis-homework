import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices

# 加载Carseats数据集
carseats = sm.datasets.get_rdataset("Carseats", package="ISLR").data
print("数据集前5行：")
print(carseats.head())

# 构建多元线性回归模型
model_formula = "Sales ~ Price + Income + Advertising + ShelveLoc"
model = sm.formula.ols(formula=model_formula, data=carseats).fit()

# 输出模型拟合报告
print("\n==========模型回归结果报告==========")
print(model.summary())

# ShelveLoc基准组
print("\n==========ShelveLoc分类变量信息==========")
print("ShelveLoc全部类别：", carseats["ShelveLoc"].unique())
print("模型生成的虚拟变量列名：", [c for c in model.params.index if "ShelveLoc" in c])

# ShelveLoc Good系数
print(f"\nShelveLoc[Good]系数值: {model.params['ShelveLoc[T.Good]']:.4f}")
print("商业含义：在Price、Income、Advertising保持不变条件下，货架位置为Good相比Bad，销售额平均增加该系数数值单位")

# 计算VIF
y, X = dmatrices(model_formula, data=carseats, return_type="dataframe")
vif_data = pd.DataFrame()
vif_data["变量名"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("\n==========各变量VIF值（方差膨胀因子）==========")
print(vif_data)
print("\nVIF判断规则：VIF>10，存在严重多重共线性风险；5~10中等；小于5基本无风险")
