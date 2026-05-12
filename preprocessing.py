import numpy as np
import pandas as pd

#CSV 읽어오기
train_df = pd.read_csv("train.csv")

def preprocess_adult_income(df):
    df = df.copy()

    #1. 결측치 처리: NaN -> 'Unknown'
    cat_missing_cols = ["workclass", "occupation", "native_country"]

    for col in cat_missing_cols:
        df[col] = df[col].replace(np.nan, "Unknown")

    #2. education 제거(education_num과 같기 때문)
    if "education" in df.columns:
        df.drop(columns=["education"], inplace=True)

    #3. capital_gain / capital_loss feature engineering(binary flag, log(1+x), net_capital)
    #binary flag
    df["has_capital_gain"] = (df["capital_gain"] > 0).astype(int)
    df["has_capital_loss"] = (df["capital_loss"] > 0).astype(int)

    #log transform
    df["log_capital_gain"] = np.log1p(df["capital_gain"])
    df["log_capital_loss"] = np.log1p(df["capital_loss"])

    #net capital
    df["net_capital"] = (df["capital_gain"] - df["capital_loss"])

    #4. native_country(United-States >> US, 나머지 >> Other)
    df["native_country"] = np.where(df["native_country"] == "United-States", "US", "Other")

    #5. race(White, 나머지 >> Non-white)
    df["race"] = np.where(df["race"] == "White", "White", "Non-white")

    #6. workclass 그룹핑
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

    #7. Feature Engineering 추가
    #is_married
    married_categories = ["Married-civ-spouse", "Married-AF-spouse"]

    df["is_married"] = (df["marital_status"].isin(married_categories).astype(int))

    #married_male
    df["married_male"] = ((df["is_married"] == 1) & (df["sex"] == "Male")).astype(int)

    #age * education_num
    df["age_education"] = (df["age"] * df["education_num"])

    #age^2
    df["age_squared"] = (df["age"] ** 2)

    #overtime
    df["overtime"] = (df["hours_per_week"] > 40).astype(int)

    #education_num * hours_per_week
    df["edu_hours"] = (df["education_num"] * df["hours_per_week"])

    #is_part_time
    df["is_part_time"] = (df["hours_per_week"] < 35).astype(int)

    #hours_per_week * age
    df["hours_age"] = (df["hours_per_week"] * df["age"])

    #capital_gain == 99999 top-coding flag
    df["is_max_gain"] = (df["capital_gain"] == 99999).astype(int)

    #net_capital 3-bin (음수/0/양수)
    df["net_capital_bin"] = np.sign(df["net_capital"]).astype(int)

    #8. occupation 그룹핑 (다른 열로 추가(묶었을 때 성능이 더 좋을 수도 있기 때문))
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

train_df = preprocess_adult_income(train_df)
train_df.to_csv("train_preprocessed.csv", index=False)