# Adult Income Prediction

Pattern Recognition 프로젝트 — UCI Adult Census 데이터셋으로 연소득 >$50K 여부 예측.

**평가 메트릭**: (F1 score + AUC) / 2

## 파일 구조

| 파일 | 설명 |
|------|------|
| `preprocessing.py` | 공통 전처리 + Feature Engineering |
| `OHE.py` | LR용 One-Hot Encoding |
| `팀원_공유사항.md` | 모델별 전처리 가이드 |

## 사용법

```python
from preprocessing import preprocess_adult_income
import pandas as pd

train_df = pd.read_csv("train.csv", na_values=['', ' '])
train_df = preprocess_adult_income(train_df)

test_df = pd.read_csv("test.csv", na_values=['', ' '])
test_df = preprocess_adult_income(test_df)
```

## 전처리 요약

1. 결측치 → "Unknown" 카테고리
2. education 제거 (education_num과 중복)
3. capital_gain/loss → binary flag, log1p, net_capital 등 파생
4. native_country → US / Other
5. race → White / Non-white
6. workclass → 4그룹 (Government / Self-employed / Private / Not-working)
7. Feature Engineering (is_married, married_male, age_squared, 상호작용 변수 등 17개)
8. occupation → 4그룹 (White-collar / Blue-collar / Service / Other), 원본 유지

모델별 인코딩/스케일링은 `팀원_공유사항.md` 참고.

## 제출 형식

`prediction.csv` — id, y_cls (0 or 1), y_prob (>50K 확률)

## 일정

- Interim: 5/17 23:59 — prediction.csv
- Final: 5/31 23:59 — code.ipynb + prediction.csv + report.pdf
