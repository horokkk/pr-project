from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# 범주형 Label Encoding
cat_cols = ['workclass', 'marital_status', 'occupation', 'relationship',
            'race', 'sex', 'native_country', 'occupation_group']

le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le  # test 변환용으로 보관

# 스케일링 X — Tree 모델은 불필요
# class_weight='balanced'로 불균형 처리
model = DecisionTreeClassifier(class_weight='balanced', random_state=42)