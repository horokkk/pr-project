# Pattern Recognition 프로젝트 계획

## 한 줄 요약
UCI Adult Income 데이터로 연소득 >$50K 여부 예측. 메트릭 = (F1 + AUC) / 2.

---

## 전체 흐름 (뭘 해야 하는지)

### Step 1: EDA (데이터 탐색) — ~2일
뭐하는 건지: 데이터가 어떻게 생겼는지 파악

- 각 컬럼 타입 확인 (숫자형 vs 범주형)
- 타겟(income) 분포 확인 — 불균형 여부 (보통 75:25 정도)
- 결측치 어디에 얼마나? (workclass, occupation, native_country에 주로 있음)
- 숫자형: 히스토그램, 박스플롯
- 범주형: value_counts, 타겟 대비 비율
- 상관관계: correlation heatmap
- 이상치 확인 (capital_gain/loss에 극단값 많음)

**산출물**: EDA 시각화 + 인사이트 정리

### Step 2: 전처리 (Preprocessing) — ~2일
뭐하는 건지: 모델에 넣을 수 있는 형태로 데이터 정리

- 결측치 처리: 최빈값 대체 or 별도 카테고리('Unknown')
- 범주형 인코딩: Label Encoding or One-Hot Encoding
  - education과 education_num 중복 → 하나 제거 고려
- 숫자형 스케일링: StandardScaler or MinMaxScaler (모델에 따라)
- feature engineering (선택):
  - capital_gain - capital_loss = net_capital
  - age 구간화 (binning)
  - hours_per_week 구간화
- train/val split: stratified 80:20 or 5-fold CV

**산출물**: 전처리 파이프라인 코드

### Step 3: 모델 학습 — ~4일
뭐하는 건지: 여러 모델 돌려보고 비교

최소 3~4개 모델 비교:

| 모델 | 난이도 | 특징 |
|------|--------|------|
| **Logistic Regression** | 쉬움 | baseline, 해석 쉬움 |
| **Random Forest** | 보통 | feature importance 제공 |
| **XGBoost / LightGBM** | 보통 | 보통 tabular 최강, 이걸로 최종 제출 |
| **SVM** | 보통 | 수업에서 배운 내용 |
| **(선택) MLP / Neural Net** | 어려움 | 딥러닝 시도 |

각 모델별로:
1. 기본 하이퍼파라미터로 학습
2. cross-validation (5-fold) 으로 성능 측정
3. 하이퍼파라미터 튜닝 (GridSearch or Optuna)
4. F1 + AUC 기록

**산출물**: 모델별 성능 비교표

### Step 4: 분석 & 해석 — ~2일
뭐하는 건지: 왜 이 모델이 좋은지, 어떤 feature가 중요한지

- Feature importance (RF, XGBoost)
- Confusion matrix
- ROC curve, PR curve
- 오분류 분석 (어떤 사람이 틀리는지)
- (선택) SHAP values
- (선택) Fairness: sex, race별 성능 차이
- (선택) Threshold tuning (0.5 말고 최적 threshold 찾기)

**산출물**: 시각화 + 해석 텍스트

### Step 5: 최종 정리 — ~3일
- prediction.csv 생성 (최종 모델로 test set 예측)
- report.pdf 작성
- code.ipynb 정리 (재현 가능하게)

---

## 타임라인

```
5/4 (일)  ← 오늘, 계획 수립 + 역할 분담
5/5~6     Step 1: EDA
5/7~8     Step 2: 전처리
5/9~13    Step 3: 모델 학습 + 튜닝
5/14~15   Step 4: 분석
5/16      prediction.csv 생성 + 검수
5/17 (일) ← Interim 제출 (prediction.csv)
5/18~25   모델 개선 + 추가 분석 + 보고서 작성
5/26~30   보고서 마무리 + 코드 정리
5/31 (일) ← 최종 제출
```

---

## 역할 분담 (안) — 3인 기준

### 방법 A: 단계별 분담 (추천)

| 역할 | 담당 | 작업 |
|------|------|------|
| **A: 데이터 + 전처리** | 팀원 1 | EDA, 결측치 처리, 인코딩, feature engineering, 전처리 파이프라인 |
| **B: 모델링** | 팀원 2 | 3~4개 모델 구현, CV, 하이퍼파라미터 튜닝, 성능 비교 |
| **C: 분석 + 보고서** | 팀원 3 (고학번=나) | feature importance, 오분류 분석, fairness, 보고서 작성, 코드 통합 |

- A가 전처리된 데이터 넘기면 → B가 모델 돌리고 → C가 분석 + 보고서
- 고학번이 C 맡으면: 전체 흐름 관리 + 보고서 품질 관리 + 코드 통합

### 방법 B: 모델별 분담

| 역할 | 담당 | 작업 |
|------|------|------|
| 팀원 1 | EDA + Logistic Regression + SVM | 전처리 공통 코드 작성 |
| 팀원 2 | Random Forest + XGBoost | 하이퍼파라미터 튜닝 |
| 팀원 3 (나) | LightGBM + 앙상블 + 보고서 | 최종 모델 선정, 보고서, 코드 통합 |

→ 전처리는 공통으로 먼저 합의하고, 각자 모델 돌리기

---

## 제출 체크리스트

- [ ] `code.ipynb` — 처음부터 끝까지 Run All 했을 때 prediction.csv 재현되는지
- [ ] `prediction.csv` — id, y_cls, y_prob 3개 컬럼
- [ ] `report.pdf` — EDA, 전처리, 모델 비교, 해석, 기여도 명시
- [ ] report에 각 팀원 contribution 명시 (필수)

---

## 참고: Adult Income 데이터 팁

- **가장 중요한 feature들**: education_num, marital_status, capital_gain, age, hours_per_week
- **class imbalance**: ~75% <=50K, ~25% >50K → F1에 영향
- **education vs education_num**: 같은 정보, 하나 제거
- **capital_gain/loss**: 대부분 0, 극단값이 있어서 log 변환 or binning 효과적
- **XGBoost/LightGBM이 보통 이 데이터에서 최강** — AUC 0.92+ 가능
