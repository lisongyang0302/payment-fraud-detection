# -*- coding: utf-8 -*-
"""step2b 模型对比：逻辑回归 vs LightGBM（不平衡学习 + 成本视角）

- 为什么反欺诈要对比树模型：工业界反欺诈主流是 GBDT 系（LightGBM/XGBoost），
  只用线性模型会显得技术面窄；但也要证明"不是无脑堆模型"——对比才有结论。
- 树模型不需要标准化：分裂点比较的是特征值大小，与尺度无关（跟 LR 的本质区别之一）。
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import lightgbm as lgb

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE, "outputs", "clean_data.csv"))

X, y = df.drop(columns=["Class"]).values, df["Class"].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

sc = StandardScaler().fit(Xtr)  # 仅训练集上 fit，防数据泄漏
Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

# 基线：逻辑回归（class_weight 处理 1:578 不平衡）
lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42).fit(Xtr_s, ytr)
p_lr = lr.predict_proba(Xte_s)[:, 1]

# LightGBM：树模型天然捕捉非线性与特征交互，class_weight 同样适用
lgbm = lgb.LGBMClassifier(
    n_estimators=500, learning_rate=0.05, num_leaves=31,
    class_weight="balanced", random_state=42, verbose=-1,
).fit(Xtr, ytr)
p_lgb = lgbm.predict_proba(Xte)[:, 1]

def metrics(p):
    return dict(
        AUC=round(roc_auc_score(yte, p), 4),
        AUPRC=round(average_precision_score(yte, p), 4),
    )

m_lr, m_lgb = metrics(p_lr), metrics(p_lgb)
print("LR 基线      :", m_lr)
print("LightGBM     :", m_lgb)

# 同一阈值 0.7 下的业务口径（挽回率/误拦率），两个模型横向比
def biz(p, t=0.7):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    return round(tp / (tp + fn) * 100, 1), round(fp / (fp + tn) * 100, 3)

biz_lr, biz_lgb = biz(p_lr), biz(p_lgb)
print(f"阈值0.7 LR   : 挽回率{biz_lr[0]}% 误拦率{biz_lr[1]}%")
print(f"阈值0.7 LGB  : 挽回率{biz_lgb[0]}% 误拦率{biz_lgb[1]}%")

np.save(os.path.join(BASE, "outputs", "p_lgb.npy"), p_lgb)
best = max([("LR", m_lr, biz_lr), ("LightGBM", m_lgb, biz_lgb)], key=lambda x: x[1]["AUPRC"])
print(f"结论：AUPRC 更优的是 {best[0]}（{best[1]['AUPRC']} vs 另一模型）")
