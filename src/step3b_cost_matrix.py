# -*- coding: utf-8 -*-
"""step3b 成本矩阵：漏放=损失该笔金额（欧元），误拦=人工审核5欧。按总成本最小化选阈值。"""
import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = np.load(os.path.join(BASE, "outputs", "p_lgb.npy"))
y = np.load(os.path.join(BASE, "outputs", "yte.npy"))
amt = np.load(os.path.join(BASE, "outputs", "amount_te.npy"))
C_REVIEW = 5.0

rows = []
for t in np.linspace(0.01, 0.99, 99):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    rows.append((round(float(t), 2), int(tp), int(fp), int(fn),
                 round(float(amt[fn].sum()), 0), round(float(amt[fn].sum() + fp * C_REVIEW), 0)))

res = pd.DataFrame(rows, columns=["threshold", "TP", "FP", "FN", "漏损欧", "总成本欧"])
os.makedirs(os.path.join(BASE, "outputs"), exist_ok=True)
res.to_csv(os.path.join(BASE, "outputs", "03b_cost_matrix.csv"), index=False)

best = res.loc[res["总成本欧"].idxmin()]
default = res[res["threshold"] == 0.5]["总成本欧"].values[0]
print(f"成本最优阈值 {best['threshold']}：总成本 {best['总成本欧']} 欧（默认0.5为 {default} 欧）")
print("成本假设：漏放=损失全额（欧元），误拦=人工审核 5 欧/笔")
