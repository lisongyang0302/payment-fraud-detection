# -*- coding: utf-8 -*-
"""step2c 时间外推(OOT)验证：按Time切分（前70%训练/后30%测试=全未来交易）

对比随机切分：LightGBM 在时间外推下 AUPRC 0.820→0.080（崩盘），LR 0.688→0.767（稳健）。
【教学】这是反欺诈模型的核心风险——欺诈模式随时间漂移，树模型对近期分布过拟合，
未来期外推能力弱；线性模型漂移钝感。业务含义：反欺诈模型需高频重训+漂移监控。
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE, "outputs", "clean_data.csv")).sort_values("Time").reset_index(drop=True)
cut = int(len(df) * 0.7)
tr, te = df.iloc[:cut], df.iloc[cut:]
X_cols = [c for c in df.columns if c != "Class"]
sc = StandardScaler().fit(tr[X_cols])
tr_s, te_s = sc.transform(tr[X_cols]), sc.transform(te[X_cols])

lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42).fit(tr_s, tr["Class"])
p = lr.predict_proba(te_s)[:, 1]
print(f"[OOT-LR ] AUC={roc_auc_score(te['Class'], p):.4f} AUPRC={average_precision_score(te['Class'], p):.4f}")

spw = (tr["Class"] == 0).sum() / (tr["Class"] == 1).sum()
gb = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                        scale_pos_weight=spw, random_state=42, verbose=-1).fit(tr[X_cols], tr["Class"])
p2 = gb.predict_proba(te[X_cols])[:, 1]
print(f"[OOT-LGB] AUC={roc_auc_score(te['Class'], p2):.4f} AUPRC={average_precision_score(te['Class'], p2):.4f}")
