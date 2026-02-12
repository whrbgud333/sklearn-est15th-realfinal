
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Data Path
data_path = r'c:\Users\User\Desktop\github\datascience\scikit-learn\data\titanic\train.csv'
output_model_path = r'c:\Users\User\Desktop\github\webML\titanic_voting_model.pkl'

print(f"Loading data from {data_path}")
try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"Error: File not found at {data_path}")
    exit(1)

# Preprocessing
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = 'Survived'

X = df[features]
y = df[target]

# Numeric: Age, SibSp, Parch, Fare
numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical: Pclass, Sex, Embarked
# Note: Pclass is ordinal but often treated as categorical. I'll treat it as categorical (one-hot) for safety or numeric. 
# Plan said Encode categorical variables (Sex, Embarked). Pclass is numeric in CSV. 
# I will include Pclass in categorical as it is a class (1, 2, 3).
categorical_features = ['Pclass', 'Sex', 'Embarked']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Models
clf1 = LogisticRegression(random_state=42, max_iter=1000)
clf2 = RandomForestClassifier(n_estimators=100, random_state=42)
clf3 = SVC(probability=True, random_state=42)
clf4 = KNeighborsClassifier()
clf5 = GradientBoostingClassifier(random_state=42)

eclf = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('svc', clf3), ('knn', clf4), ('gb', clf5)],
    voting='soft'
)

# Pipeline
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', eclf)])

# Train
print("Training model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

# Evaluate
y_pred = model_pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {acc:.4f}')

# Save
joblib.dump(model_pipeline, output_model_path)
print(f"Model saved to {output_model_path}")
