# Adult Income Prediction

Pattern Recognition 프로젝트 — UCI Adult Census 데이터셋으로 연소득 >$50K 여부 예측.

## 평가 메트릭

**(F1 score + AUC) / 2**

## 파일 구조

```
pr-project/
├── preprocessing.py       # 공통 전처리 + Feature Engineering
├── OHE.py                 # LR용 One-Hot Encoding
├── 팀원_공유사항.md         # 모델별 전처리 가이드
└── README.md
```

## 사용법

### 1. 공통 전처리

```python
from preprocessing import preprocess_adult_income
import pandas as pd

# train
train_df = pd.read_csv("train.csv", na_values=['', ' '])
train_df = preprocess_adult_income(train_df)

# test (동일 함수 적용)
test_df = pd.read_csv("test.csv", na_values=['', ' '])
test_df = preprocess_adult_income(test_df)
```

출력: `train_preprocessed.csv`

### 2. 모델별 인코딩

| 모델 | 인코딩 | 스케일링 | capital 원본 |
|------|--------|---------|-------------|
| Logistic Regression | OHE (`drop_first=True`) | StandardScaler | drop |
| Decision Tree | Label Encoding | X | 유지 |
| Random Forest | Label Encoding | X | 유지 |
| LightGBM | Native Categorical / Target Encoding | X | 유지 |

자세한 코드는 `팀원_공유사항.md` 참고.

## 전처리 파이프라인

### Step 1~6: 데이터 정리

| 단계 | 내용 |
|------|------|
| 1 | 결측치 → "Unknown" 카테고리 |
| 2 | education 제거 (education_num과 중복) |
| 3 | capital_gain/loss → binary flag + log1p + net_capital |
| 4 | native_country → US / Other |
| 5 | race → White / Non-white |
| 6 | workclass → Government / Self-employed / Private / Not-working |

### Step 7: Feature Engineering (17개 파생 변수)

| 변수 | 수식 |
|------|------|
| has_capital_gain | capital_gain > 0 |
| has_capital_loss | capital_loss > 0 |
| log_capital_gain | log(1 + capital_gain) |
| log_capital_loss | log(1 + capital_loss) |
| net_capital | capital_gain - capital_loss |
| is_married | marital_status ∈ {Married-civ-spouse, Married-AF-spouse} |
| married_male | is_married & (sex == Male) |
| is_spouse | relationship ∈ {Husband, Wife} |
| age_education | age x education_num |
| age_squared | age^2 |
| overtime | hours_per_week > 40 |
| edu_hours | education_num x hours_per_week |
| is_part_time | hours_per_week < 35 |
| hours_age | hours_per_week x age |
| is_max_gain | capital_gain == 99999 |
| net_capital_bin | sign(net_capital) |
| occupation_group | occupation → 4그룹 (Step 8) |

### Step 8: occupation 그룹핑

| 그룹 | 직업 |
|------|------|
| White-collar | Exec-managerial, Prof-specialty, Adm-clerical, Tech-support, Sales |
| Blue-collar | Craft-repair, Machine-op-inspct, Transport-moving, Handlers-cleaners, Farming-fishing |
| Service | Other-service, Priv-house-serv, Protective-serv |
| Other | Armed-Forces, Unknown |

원본 `occupation` 컬럼은 유지 (모델별 인코딩용).

## 제출 형식

```
prediction.csv
├── id       : 샘플 식별자
├── y_cls    : 0 (<=50K) 또는 1 (>50K)
└── y_prob   : >50K 예측 확률
```

## 일정

- Interim: 5/17 (일) 23:59 — prediction.csv
- Final: 5/31 (일) 23:59 — code.ipynb + prediction.csv + report.pdf
