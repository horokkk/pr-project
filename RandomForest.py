import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

from preprocessing import preprocess_adult_income

# test 전처리
test_df = pd.read_csv("test.csv")
test_df = preprocess_adult_income(test_df)

df = pd.read_csv("train_preprocessed.csv")

# target encording
df["income"] = df["income"].apply(lambda x: 1 if ">50K" in x else 0)

# id 보관 및 제거
ids = df["id"]
df.drop(columns=["id"], inplace=True)

# X, y 분리
y = df["income"]
X = df.drop(columns=["income"])

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Label Encoding
cat_cols = ['workclass', 'marital_status', 'occupation', 'relationship',
            'race', 'sex', 'native_country', 'occupation_group']

le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le

# # RandomForest 모델
# model = RandomForestClassifier(
#     n_estimators=500,
#     class_weight='balanced',
#     random_state=42
# )

# 하이퍼파라미터 튜닝
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# test 적용
# test_df = pd.read_csv("test_preprocessed.csv")  # preprocessing.py 적용된 것
for col in cat_cols:
    test_df[col] = le_dict[col].transform(test_df[col])

# 성능 측정
from sklearn.model_selection import cross_val_score, StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
auc_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
print(f"F1: {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")
print(f"AUC: {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")


# 모델 학습
model.fit(X, y)
X_test = test_df.drop(columns=["id"])

# test 예측 확률 저장
y_prob = model.predict_proba(X_test)[:, 1]
np.save('prob_rf.npy', y_prob)