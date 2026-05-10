import numpy as np
import pandas as pd

#CSV 읽어오기
df = pd.read_csv("train_preprocessed.csv")

#타겟 변수 수치화
if df["income"].dtype == 'object': df["income"] = df["income"].apply(lambda x : 1 if ">50K" in x else 0)

#범주형 변수 수치화 (OHE)
target_ohe_cols = ['workclass', 'marital_status', 'relationship', 'race', 'sex', 'native_country', 'occupation_group']
train_final = pd.get_dummies(df, columns=target_ohe_cols, drop_first=True, dtype=int)

train_final.to_csv("train_preprocessed_OHE.csv", index=False)