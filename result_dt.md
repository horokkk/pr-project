# Decision Tree 모델 결과 분석

## 1. 모델 개요

- 모델: Decision Tree
- 검증 방법: Stratified 5-Fold Cross Validation
- 사용 데이터: `train_preprocessed.csv`

---

# 2. 모델 성능 결과

## F1 Score

```text
F1: 0.6766 ± 0.0074
```

### 해석

F1 Score: Precision(정밀도)와 Recall(재현율)의 조화 평균

표준편차가 0.0074로 낮아 Fold 간 성능 차이가 크지 않음.

---

## AUC Score

```text
AUC: 0.9059 ± 0.0035
```

### Fold별 AUC

```text
[0.9075794  0.89992368 0.90481437 0.90688988 0.91041143]
```

### 해석

AUC(Area Under Curve): 모델이 고소득자와 저소득자를 얼마나 잘 구분하는지를 나타내는 지표

본 모델의 AUC는 0.9059로 매우 높은 수준의 분류 성능을 보임. >> 고소득자와 저소득자의 상대적 순위를 잘 구분하고 있음을 의미.

표준편차가 0.0035로 매우 낮아, Fold 간 성능 차이가 크지 않음.

### 기본 모델과 튜닝 모델 성능 비교

| Model | Mean F1 | Mean AUC | F1 Std | AUC Std |
|---|---|---|---|---|
| Default DT | 0.612309 | 0.767276 | 0.003594 | 0.002494 |
| Tuned DT | 0.676607 | 0.905924 | 0.007445 | 0.003494 |

튜닝 모델이 더 성능이 좋은 것을 알 수 있음.

---

# 3. Feature Importance 분석

## Top 10 Feature Importance

| Feature | Importance |
|---|---|
| is_spouse | 0.521185 |
| age_education | 0.127517 |
| capital_gain | 0.126136 |
| edu_hours | 0.088618 |
| occupation_group | 0.048495 |
| net_capital | 0.023053 |
| capital_loss | 0.016353 |
| log_capital_gain | 0.015558 |
| hours_age | 0.006938 |
| education_num | 0.005188 |

---

# 4. Feature Engineering 효과 분석

전처리 과정에서 추가된 파생변수:

- has_capital_gain, has_capital_loss
- log_capital_gain, log_capital_loss
- net_capital
- is_married
- married_male
- is_spouse
- age_education
- age_squared
- overtime
- edu_hours
- is_part_time
- hours_age
- is_max_gain
- net_capital_bin

Top 10 Feature Importance를 보면 원본 변수보다 파생변수들의 중요도가 높게 나타남. >> Feature Engineering이 Income Classification 성능 향상에 효과적으로 기여함.

특히:

- is_spouse
- age_education
- edu_hours

등의 파생변수가 높은 importance를 기록함.

---

# 5. 예측 확률 결과

```text
[0.03015321 0.65404539 0.16785674 ... 1. 1. 0.56198918]
```

---

# 6. 종합 평가

| 항목 | 평가 |
|---|---|
| 전처리 | 우수 |
| Feature Engineering | 효과적 |
| 모델 안정성 | 높음 |
| AUC 성능 | 매우 우수 |
| F1 성능 | 준수 |
| 앙상블 활용성 | 높음 |