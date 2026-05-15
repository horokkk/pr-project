import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

df = pd.read_csv("train_preprocessed.csv")

#target encoding
df["income"] = df["income"].apply(lambda x : 1 if ">50K" in x else 0)

#id 보관 및 제거
ids = df["id"]
df.drop(columns=["id"], inplace=True)

#X, y 분리
y = df["income"]
X = df.drop(columns=["income"])

# 범주형 Label Encoding
cat_cols = ['workclass', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country', 'occupation_group']

le_dict = {}

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le  #test 변환용으로 보관

#기본 Decision Tree 모델
#model = DecisionTreeClassifier(class_weight='balanced', random_state=42)

#하이퍼파라미터 튜닝
model = DecisionTreeClassifier(max_depth=8, min_samples_split=20, min_samples_leaf=10, criterion='gini', class_weight='balanced', random_state=42)

#성능 측정
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
auc_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')

print(f"F1: {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")
print(f"AUC: {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")

#ROC-AUC 평가
auc_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')

print("=" * 50)
print("Decision Tree AUC Scores")
print(auc_scores)
print(f"Mean AUC : {auc_scores.mean():.4f}")
print(f"Std AUC  : {auc_scores.std():.4f}")

#전체 train 데이터로 학습
model.fit(X, y)

#Feature Importance 출력
importance_df = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
importance_df = importance_df.sort_values(by="importance", ascending=False)

print("=" * 50)
print("Top 10 Feature Importance")
print(importance_df.head(10))