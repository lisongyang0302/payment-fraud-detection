# -*- coding: utf-8 -*-
"""step2 建模：标准化防泄漏 + class_weight 处理 1:578 不平衡 + AUC/AUPRC 评估"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE, "outputs", "clean_data.csv"))

X, y = df.drop(columns=["Class"]).values, df["Class"].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

sc = StandardScaler().fit(Xtr)  # 仅在训练集上 fit，防数据泄漏
Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42).fit(Xtr_s, ytr)
p = lr.predict_proba(Xte_s)[:, 1]
print(f"AUC={roc_auc_score(yte, p):.4f}  AUPRC={average_precision_score(yte, p):.4f}")

os.makedirs(os.path.join(BASE, "outputs"), exist_ok=True)
np.save(os.path.join(BASE, "outputs", "p_bal.npy"), p)
np.save(os.path.join(BASE, "outputs", "yte.npy"), yte)
np.save(os.path.join(BASE, "outputs", "amount_te.npy"), Xte[:, list(df.columns).index("Amount")])
