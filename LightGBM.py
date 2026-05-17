"""
LightGBM - Adult Income Prediction
Pattern Recognition Project (Interim: 5/17)
Metric: (F1 + AUC) / 2

Colab 사용법:
1. train.csv, test.csv 업로드
2. 첫 셀: !pip install lightgbm optuna
3. 이 코드 전체 실행
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. 전처리 함수 (preprocessing.py 인라인)
# ============================================================
def preprocess_adult_income(df):
    df = df.copy()

    # 결측치 처리: NaN -> 'Unknown'
    cat_missing_cols = ["workclass", "occupation", "native_country"]
    for col in cat_missing_cols:
        df[col] = df[col].replace(np.nan, "Unknown")

    # education 제거 (education_num과 같기 때문)
    if "education" in df.columns:
        df.drop(columns=["education"], inplace=True)

    # capital_gain / capital_loss feature engineering
    df["has_capital_gain"] = (df["capital_gain"] > 0).astype(int)
    df["has_capital_loss"] = (df["capital_loss"] > 0).astype(int)
    df["log_capital_gain"] = np.log1p(df["capital_gain"])
    df["log_capital_loss"] = np.log1p(df["capital_loss"])
    df["net_capital"] = df["capital_gain"] - df["capital_loss"]

    # native_country (US vs Other)
    df["native_country"] = np.where(df["native_country"] == "United-States", "US", "Other")

    # race (White vs Non-white)
    df["race"] = np.where(df["race"] == "White", "White", "Non-white")

    # workclass 그룹핑
    gov = ["Federal-gov", "Local-gov", "State-gov"]
    self_emp = ["Self-emp-inc", "Self-emp-not-inc"]

    def simplify_workclass(x):
        if x in gov:
            return "Government"
        elif x in self_emp:
            return "Self-employed"
        elif x == "Private":
            return "Private"
        else:
            return "Not-working"

    df["workclass"] = df["workclass"].apply(simplify_workclass)

    # Feature Engineering
    married_categories = ["Married-civ-spouse", "Married-AF-spouse"]
    df["is_married"] = df["marital_status"].isin(married_categories).astype(int)
    df["married_male"] = ((df["is_married"] == 1) & (df["sex"] == "Male")).astype(int)
    df["is_spouse"] = df["relationship"].isin(["Husband", "Wife"]).astype(int)
    df["age_education"] = df["age"] * df["education_num"]
    df["age_squared"] = df["age"] ** 2
    df["overtime"] = (df["hours_per_week"] > 40).astype(int)
    df["edu_hours"] = df["education_num"] * df["hours_per_week"]
    df["is_part_time"] = (df["hours_per_week"] < 35).astype(int)
    df["hours_age"] = df["hours_per_week"] * df["age"]
    df["is_max_gain"] = (df["capital_gain"] == 99999).astype(int)
    df["net_capital_bin"] = np.sign(df["net_capital"]).astype(int)

    # occupation 그룹핑
    white_collar = ["Exec-managerial", "Prof-specialty", "Adm-clerical", "Tech-support", "Sales"]
    blue_collar = ["Craft-repair", "Machine-op-inspct", "Transport-moving", "Handlers-cleaners", "Farming-fishing"]
    service = ["Other-service", "Priv-house-serv", "Protective-serv"]

    def simplify_occupation(x):
        if x in white_collar:
            return "White-collar"
        elif x in blue_collar:
            return "Blue-collar"
        elif x in service:
            return "Service"
        else:
            return "Other"

    df["occupation_group"] = df["occupation"].apply(simplify_occupation)

    return df

# ============================================================
# 2. 데이터 로드 + 전처리
# ============================================================
train_df = pd.read_csv("train.csv", na_values=["", " "])
test_df = pd.read_csv("test.csv", na_values=["", " "])

train_df = preprocess_adult_income(train_df)
test_df = preprocess_adult_income(test_df)

# 타겟 수치화
train_df["income"] = train_df["income"].apply(lambda x: 1 if ">50K" in str(x) else 0)

# id 분리
train_ids = train_df["id"]
test_ids = test_df["id"]
train_df.drop(columns=["id"], inplace=True)
test_df.drop(columns=["id"], inplace=True)

# X, y 분리
y = train_df["income"]
X = train_df.drop(columns=["income"])
X_test = test_df.copy()

# ============================================================
# 3. LightGBM Native Categorical 설정
# ============================================================
cat_cols = ["workclass", "marital_status", "occupation", "relationship",
            "race", "sex", "native_country", "occupation_group"]

for col in cat_cols:
    X[col] = X[col].astype("category")
    X_test[col] = X_test[col].astype("category")

print(f"Train shape: {X.shape}")
print(f"Test shape: {X_test.shape}")
print(f"Target distribution: {y.value_counts().to_dict()}")
print(f"Categorical features: {cat_cols}")
print()

# ============================================================
# 4. 평가 함수
# ============================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_lgbm(params, X, y, skf):
    """5-Fold CV로 F1, AUC 측정"""
    oof_prob = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
        dval = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        oof_prob[val_idx] = model.predict(X_val)

    oof_cls = (oof_prob >= 0.5).astype(int)
    f1 = f1_score(y, oof_cls)
    auc = roc_auc_score(y, oof_prob)
    return f1, auc, oof_prob

# ============================================================
# 5. Baseline 평가
# ============================================================
baseline_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1,
    "seed": 42,
    "n_jobs": -1,
    "is_unbalance": True,
}

print("=" * 50)
print("Baseline (default params)")
print("=" * 50)
f1_base, auc_base, _ = evaluate_lgbm(baseline_params, X, y, skf)
print(f"F1:  {f1_base:.4f}")
print(f"AUC: {auc_base:.4f}")
print(f"(F1+AUC)/2: {(f1_base + auc_base) / 2:.4f}")
print()

# ============================================================
# 6. Optuna 하이퍼파라미터 튜닝
# ============================================================
def objective(trial):
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "seed": 42,
        "n_jobs": -1,
        "is_unbalance": True,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
    }

    f1, auc, _ = evaluate_lgbm(params, X, y, skf)
    return (f1 + auc) / 2


print("=" * 50)
print("Optuna Tuning (100 trials)")
print("=" * 50)

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=100, show_progress_bar=True)

best_params = study.best_params
best_params.update({
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1,
    "seed": 42,
    "n_jobs": -1,
    "is_unbalance": True,
})

print(f"\nBest (F1+AUC)/2: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
print()

# ============================================================
# 7. 최종 모델 + Threshold Tuning
# ============================================================
print("=" * 50)
print("Final Model (tuned params)")
print("=" * 50)

f1_final, auc_final, oof_prob = evaluate_lgbm(best_params, X, y, skf)
print(f"F1:  {f1_final:.4f}")
print(f"AUC: {auc_final:.4f}")
print(f"(F1+AUC)/2: {(f1_final + auc_final) / 2:.4f}")
print()

# Threshold tuning (F1 최적화)
thresholds = np.arange(0.3, 0.7, 0.01)
best_thr, best_f1_thr = 0.5, f1_final
for thr in thresholds:
    f1_t = f1_score(y, (oof_prob >= thr).astype(int))
    if f1_t > best_f1_thr:
        best_f1_thr = f1_t
        best_thr = thr

print(f"Optimal threshold: {best_thr:.2f} (F1: {best_f1_thr:.4f})")
auc_final_thr = roc_auc_score(y, oof_prob)
print(f"Final (F1+AUC)/2 with threshold tuning: {(best_f1_thr + auc_final_thr) / 2:.4f}")
print()

# ============================================================
# 8. Test 예측 + prediction.csv 생성
# ============================================================
print("=" * 50)
print("Test Prediction")
print("=" * 50)

test_probs = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
    dval = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols)

    model = lgb.train(
        best_params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    test_probs += model.predict(X_test) / 5

# prediction.csv
prediction = pd.DataFrame({
    "id": test_ids,
    "y_cls": (test_probs >= best_thr).astype(int),
    "y_prob": test_probs
})

prediction.to_csv("prediction.csv", index=False)
print(f"prediction.csv saved! ({len(prediction)} rows)")
print(f"  y_cls distribution: {prediction['y_cls'].value_counts().to_dict()}")
print(f"  y_prob mean: {prediction['y_prob'].mean():.4f}")

# 팀 공유용
np.save("prob_lgbm.npy", test_probs)
print("prob_lgbm.npy saved!")
print()

# ============================================================
# 9. Feature Importance (보고서용)
# ============================================================
print("=" * 50)
print("Top 15 Feature Importance (gain)")
print("=" * 50)

importance = pd.DataFrame({
    "feature": model.feature_name(),
    "importance": model.feature_importance(importance_type="gain")
}).sort_values("importance", ascending=False)

for i, row in importance.head(15).iterrows():
    print(f"  {row['feature']:25s} {row['importance']:.1f}")

print("\nDone!")
