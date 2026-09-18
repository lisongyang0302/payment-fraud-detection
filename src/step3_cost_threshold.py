# -*- coding: utf-8 -*-
"""step3 策略落地：阈值扫描——欺诈挽回率 vs 误拦率 的业务权衡"""
import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = np.load(os.path.join(BASE, "outputs", "p_bal.npy"))
y = np.load(os.path.join(BASE, "outputs", "yte.npy"))
amt = np.load(os.path.join(BASE, "outputs", "amount_te.npy"))

rows = []
for t in np.linspace(0.01, 0.99, 99):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    rows.append({
        "threshold": round(float(t), 2),
        "TP": int(tp), "FP": int(fp), "FN": int(fn),
        "欺诈挽回率%": round(tp / (tp + fn) * 100, 1),
        "误拦率%": round(fp / (fp + tn) * 100, 3),
        "漏损金额": round(float(amt[fn].sum()), 0),
    })

res = pd.DataFrame(rows)
os.makedirs(os.path.join(BASE, "outputs"), exist_ok=True)
res.to_csv(os.path.join(BASE, "outputs", "03_threshold_sweep.csv"), index=False)

for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
    r_ = res.iloc[(res["threshold"] - t).abs().argmin()]
    print(f"阈值{r_['threshold']}: 挽回率{r_['欺诈挽回率%']}% | 误拦率{r_['误拦率%']}% | 漏损{r_['漏损金额']}元")
print("结论：阈值=策略旋钮——调低多拦(误拦成本升)，调高少拦(漏损升)，按业务成本结构选点")
