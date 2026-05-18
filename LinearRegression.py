import pandas as pd

df = pd.read_csv("train_preprocessed_OHE.csv")

# 1. 타겟 수치화
df["income"] = df["income"].apply(lambda x: 1 if ">50K" in x else 0)

# 2. id 분리 (학습에서 제외, 제출용으로 보관)
ids = df["id"]
df.drop(columns=["id"], inplace=True)

# 3. X, y 분리
y = df["income"]
X = df.drop(columns=["income"])

# capital_gain/capital_loss 원본 drop (log 변환본으로 대체)
# → 99,999 같은 극단값이 StandardScaler 후에도 왜곡을 일으킴
X.drop(columns=["capital_gain", "capital_loss"], inplace=True)