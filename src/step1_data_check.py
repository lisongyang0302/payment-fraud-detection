# -*- coding: utf-8 -*-
"""step1 数据体检：体量/不平衡度/缺失重复/金额分布

数据放置：将 Kaggle 下载的 creditcard.csv 放入 ../data/ 目录即可（下载地址见 README）。
"""
import pandas as pd
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "creditcard.csv")
OUT = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)

if not os.path.exists(DATA):
    raise FileNotFoundError(
        f"未找到数据文件：{DATA}\n"
        "请从 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud 下载 "
        "creditcard.csv 并放入 data/ 目录。"
    )

df = pd.read_csv(DATA)
print(f"样本 {len(df):,} | 欺诈 {df['Class'].sum():,} | 不平衡 1:{len(df)//df['Class'].sum()}")
print(f"缺失 {df.isna().sum().sum()} | 重复 {df.duplicated().sum()}")

df = df.drop_duplicates().reset_index(drop=True)
os.makedirs(os.path.join(BASE, "outputs"), exist_ok=True)
df.to_csv(os.path.join(BASE, "outputs", "clean_data.csv"), index=False)

fa = df.loc[df.Class == 1, "Amount"]
print(f"欺诈金额: 中位数 {fa.median():.2f} | 均值 {fa.mean():.2f} | max {fa.max():.0f}")
print(f"洞察: 欺诈多为小额试刷(验证卡可用性)后再大额盗刷，中位数仅 {fa.median():.1f} 元")
