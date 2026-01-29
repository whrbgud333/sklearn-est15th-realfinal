
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import json

# Load Data
data_path = r"uploads\processed_titanic_20260124_submission.csv"
df = pd.read_csv(data_path)
target = "target"

# Encode Categorical Variables
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col].astype(str))

# Split Data
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Determine Task Type (Classification/Regression) based on target unique values
is_classification = False
if y.nunique() < 20 or y.dtype == 'object':
    is_classification = True

results = {}

if is_classification:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results['type'] = 'Classification'
    results['accuracy'] = acc
    results['model'] = 'RandomForestClassifier'
else:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results['type'] = 'Regression'
    results['mse'] = mse
    results['r2'] = r2
    results['model'] = 'RandomForestRegressor'

print(json.dumps(results))
