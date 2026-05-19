import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from preprocessing import preprocess_adult_income

df = pd.read_csv("train_preprocessed_OHE.csv")

#id 분리 (학습에서 제외, 제출용으로 보관)
ids = df["id"]
df.drop(columns=["id"], inplace=True)

#X, y 분리
y = df["income"]
X = df.drop(columns=["income"])

#occupation 제거
X.drop(columns=["occupation"], inplace=True)

#capital_gain/capital_loss 원본 drop (log 변환본으로 대체)
#→ 99,999 같은 극단값이 StandardScaler 후에도 왜곡을 일으킴
X.drop(columns=["capital_gain", "capital_loss"], inplace=True)

#수치형 columns 스케일링
num_cols = X.select_dtypes(include=["int64", "float64"]).columns

scaler = StandardScaler()

X[num_cols] = scaler.fit_transform(X[num_cols])

#기본 Linear Regression 모델
#model = LogisticRegression(max_iter=5000, class_weight='balanced', random_state=42)

#하이퍼파라미터 튜닝
model = LogisticRegression(max_iter=5000, class_weight='balanced', C=10.0, solver='lbfgs', random_state=42)

#성능 측정
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
auc_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')

print(f"F1: {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")
print(f"AUC: {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")
print(f"(F1 + AUC) / 2: {(f1_scores.mean() + auc_scores.mean()) / 2:.4f}")

#전체 train 데이터로 학습
model.fit(X, y)

#Feature Importance 출력
importance_df = pd.DataFrame({"feature": X.columns, "coef": model.coef_[0]})
importance_df["abs_coef"] = importance_df["coef"].abs()
importance_df = importance_df.sort_values(by="abs_coef", ascending=False)

print("Top 10 Feature Importance")
print(importance_df.head(10))

#============================================================================================

#test
test_df = pd.read_csv("test.csv", na_values=['', ' '])
test_df = preprocess_adult_income(test_df)
test_df.to_csv("test_preprocessed.csv", index=False)

test_df = pd.read_csv("test_preprocessed.csv")

#범주형 변수 수치화 (OHE)
target_ohe_cols = ['workclass', 'marital_status', 'relationship', 'race', 'sex', 'native_country', 'occupation_group']
test_final = pd.get_dummies(test_df, columns=target_ohe_cols, drop_first=True, dtype=int)

test_final.to_csv("test_preprocessed_OHE.csv", index=False)

test_df = pd.read_csv("test_preprocessed_OHE.csv");

test_ids = test_df["id"]
test_df.drop(columns=["id"], inplace=True)

test_df.drop(columns=["occupation"], inplace=True)

#capital_gain/capital_loss 원본 drop (log 변환본으로 대체)
test_df.drop(columns=["capital_gain", "capital_loss"], inplace=True)

#scaling
test_df[num_cols] = scaler.transform(test_df[num_cols])
X_test = test_df

#예측
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

#확률 저장
np.save("prob_lr.npy", y_prob)
