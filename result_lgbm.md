# LightGBM 결과 요약

## 모델 개요

- **모델**: LightGBM (Gradient Boosting)
- **범주형 처리**: Native Categorical (LightGBM 내장)
- **불균형 대응**: `is_unbalance=True`
- **평가**: 5-Fold Stratified CV
- **하이퍼파라미터 튜닝**: Optuna (100 trials, TPE sampler)
- **Threshold Tuning**: F1 최적 threshold 탐색 (0.30~0.70)

---

## 성능

| 단계 | F1 | AUC | (F1+AUC)/2 |
|------|:--:|:---:|:----------:|
| Baseline (default) | - | - | ~0.818 |
| Optuna 튜닝 후 | 0.7131 | 0.9274 | 0.8203 |
| + Threshold Tuning (0.66) | **0.7235** | **0.9274** | **0.8254** |

---

## 최적 하이퍼파라미터

| 파라미터 | 값 |
|---------|-----|
| learning_rate | 0.0180 |
| num_leaves | 117 |
| max_depth | 5 |
| min_child_samples | 10 |
| subsample | 0.8246 |
| colsample_bytree | 0.6867 |
| reg_alpha | 0.7283 |
| reg_lambda | 0.8655 |
| min_split_gain | 0.1560 |

### Hyperparameter Importance (Optuna)

1. **learning_rate** — 가장 큰 영향 (~0.31)
2. **reg_alpha** — ~0.20
3. **reg_lambda** — ~0.14
4. **max_depth** — ~0.10
5. **colsample_bytree** — ~0.09

---

## Confusion Matrix (threshold=0.66)

|  | Predicted <=50K | Predicted >50K |
|--|:-:|:-:|
| **Actual <=50K** | 26,766 | 2,963 |
| **Actual >50K** | 2,369 | 6,975 |

- Precision (>50K): 6975 / (6975+2963) = **0.702**
- Recall (>50K): 6975 / (6975+2369) = **0.747**

---

## Feature Importance (Top 15, Gain)

| 순위 | Feature | Gain |
|:---:|---------|-----:|
| 1 | is_married | 166,620 |
| 2 | is_spouse | 151,230 |
| 3 | age_education | 109,869 |
| 4 | capital_gain | 109,588 |
| 5 | edu_hours | 92,003 |
| 6 | occupation | 80,353 |
| 7 | relationship | 72,694 |
| 8 | hours_age | 48,111 |
| 9 | education_num | 36,908 |
| 10 | log_capital_gain | 35,288 |
| 11 | capital_loss | 34,626 |
| 12 | age | 26,450 |
| 13 | net_capital | 24,053 |
| 14 | hours_per_week | 16,259 |
| 15 | log_capital_loss | 11,472 |

---

## Test 예측 분포

- 전체: 9,769건
- <=50K: 7,317 (74.9%)
- \>50K: 2,452 (25.1%)
- y_prob 평균: 0.3372
