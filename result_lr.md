# Linear Regression 모델 결과 분석

## 1. 모델 개요

- 모델: Linear Regresion
- 검증 방법: Stratified 5-Fold Cross Validation
- 사용 데이터: `train_preprocessed_OHE.csv`

---

# 2. 모델 성능 결과

## F1 Score

```text
F1: 0.6807 ± 0.0077
```

### 해석

0.68 수준으로 AUC 대비 상대적으로 낮게 나타났다.
이는 모델의 확률 예측 능력은 우수하나, 실제 이진 분류 threshold에서 precision과 recall 균형이 완벽하지 않음을 의미한다.

표준편차는 매우 작게 나타났기 때문에 교차검증 간 fold 간 성능 편차가 작고, 모델이 안정적으로 동작하고 있음을 알 수 있다.

---

## AUC Score

```text
AUC: 0.9096 ± 0.0040
```

### 해석

AUC가 약 0.91로 매우 높은 값을 기록했다.
이는 모델이 고소득 클래스와 저소득 클래스를 전반적으로 잘 구분하고 있음을 의미한다.

예측 확률 자체의 품질은 상당히 우수한 편

F1과 마찬가지로 표준편차는 매우 작게 나타났기 때문에 교차검증 간 fold 간 성능 편차가 작고, 모델이 안정적으로 동작하고 있음을 알 수 있다.

---

# 3. Feature Importance 분석

| Feature | Importance |
|---|---|
| log_capital_gain | 가장 큰 양의 영향 |
| has_capital_gain | 강한 음의 계수 |
| has_capital_loss | 강한 음의 계수 |
| log_capital_loss | 양의 영향 |
| age | 나이가 증가할수록 고소득 가능성 증가 |

Capital 관련 feature의 영햐이 매우 크게 나타났음을 알 수 있다.
이는 자본 이득과 자본 손실 정보가 소득 예측에 매우 중요한 역할을 하였음을 의미한다.