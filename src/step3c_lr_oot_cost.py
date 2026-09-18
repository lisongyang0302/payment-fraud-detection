# -*- coding: utf-8 -*-
"""step3c 成本矩阵（时间外推口径）：LR + OOT测试集（时间后30%）的99档阈值扫描。

随机切分下 LightGBM 最优阈值 0.74（总成本31欧），但 OOT 验证显示其时间外推崩盘
（AUPRC 0.820→0.080），随机切分选点不可部署——部署口径应采用时间外推下稳健的 LR。
本脚本按同一成本假设（漏放=损失该笔金额，误拦=人工审核5欧）在 OOT 测试集上
对 LR 概率做 99 档扫描，输出可部署的最优阈值与成本。
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE, "outputs", "clean_data.csv")).sort_values("Time").reset_index(drop=True)
cut = int(len(df) * 0.7)
tr, te = df.iloc[:cut], df.iloc[cut:]
X_cols = [c for c in df.columns if c not in ("Class", "Time")]

sc = StandardScaler().fit(tr[X_cols])
lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
lr.fit(sc.transform(tr[X_cols]), tr["Class"])
p = lr.predict_proba(sc.transform(te[X_cols]))[:, 1]
y = te["Class"].values
amt = te["Amount"].values

C_REVIEW = 5.0
rows = []
for t in np.linspace(0.01, 0.99, 99):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    rows.append((round(float(t), 2), int(tp), int(fp), int(fn),
                 round(float(amt[fn].sum()), 2), round(float(amt[fn].sum() + fp * C_REVIEW), 2)))

res = pd.DataFrame(rows, columns=["threshold", "TP", "FP", "FN", "漏损欧", "总成本欧"])
res.to_csv(os.path.join(BASE, "outputs", "03c_cost_matrix_lr_oot.csv"), index=False)

best = res.loc[res["总成本欧"].idxmin()]
default = res[res["threshold"] == 0.5]["总成本欧"].values[0]
drop = (1 - best["总成本欧"] / default) * 100
print(f"测试集={len(te)}笔 欺诈={int(y.sum())}笔 基率={y.mean():.4%}")
print(f"LR+OOT 成本最优阈值 {best['threshold']}：总成本 {best['总成本欧']} 欧（默认0.5为 {default} 欧，降 {drop:.0f}%）")
print(f"最优处 TP={int(best['TP'])} FP={int(best['FP'])} FN={int(best['FN'])} 漏损={best['漏损欧']}欧")
print(f"算式：总成本 {best['总成本欧']} = 漏放 {best['漏损欧']} + 误拦 {int(best['FP'])}×5")
