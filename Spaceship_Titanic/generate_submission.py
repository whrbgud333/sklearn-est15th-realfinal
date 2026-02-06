import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
import os

# 1. Load Data
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
SUBMISSION_PATH = "submission/submission.csv"
MODEL_PATH = "AutogluonModels/ag-3600s-final"

print("Loading data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# Combine for preprocessing
train_len = len(train_df)
all_data = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)

print("Preprocessing data...")

# 2. Feature Engineering (Replicating notebook logic)

# Spending Columns
spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
all_data[spending_cols] = all_data[spending_cols].fillna(0)
all_data['TotalSpending'] = all_data[spending_cols].sum(axis=1)

# Spending Group (Quantiles)
# Note: qcut might fail if too many zeros/duplicates. Notebook used duplicates='drop'
try:
    all_data['SpendingGroup'] = pd.qcut(all_data['TotalSpending'], q=5, duplicates='drop', labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
except Exception as e:
    print(f"Warning: SpendingGroup qcut failed: {e}")

# CryoSleep Imputation
all_data.loc[(all_data['CryoSleep'].isna()) & (all_data['TotalSpending'] > 0), 'CryoSleep'] = False
all_data.loc[(all_data['CryoSleep'].isna()) & (all_data['TotalSpending'] == 0), 'CryoSleep'] = True
# Fallback for remaining CryoSleep NaNs? Notebook printed "0 missing", so this handled it likely.
# But just in case:
if all_data['CryoSleep'].isna().sum() > 0:
    all_data['CryoSleep'] = all_data['CryoSleep'].fillna(False) # Default fallback

# Age Imputation
age_median = all_data['Age'].median()
all_data['Age'] = all_data['Age'].fillna(age_median)

# Age Group
def update_age_group(age):
    if age <= 4: return 'Baby'
    elif age <= 12: return 'Child'
    elif age <= 19: return 'Teenager'
    elif age <= 40: return 'Adult'
    elif age <= 60: return 'Middle Aged'
    else: return 'Senior'

all_data['AgeGroup'] = all_data['Age'].apply(update_age_group)

# VIP Imputation
all_data.loc[(all_data['VIP'].isna()) & (all_data['TotalSpending'] == 0), 'VIP'] = False
all_data.loc[(all_data['VIP'].isna()) & (all_data['Age'] <= 19), 'VIP'] = False
all_data.loc[(all_data['VIP'].isna()) & (all_data['HomePlanet'] == 'Earth'), 'VIP'] = False
all_data['VIP'] = all_data['VIP'].fillna(False).astype(bool)

# Destination Imputation
dest_mode = all_data['Destination'].mode()[0]
all_data['Destination'] = all_data['Destination'].fillna(dest_mode)

# Group and GroupSize
all_data['Group'] = all_data['PassengerId'].str.split('_').str[0]
group_sizes = all_data.groupby('Group').size()
all_data['GroupSize'] = all_data['Group'].map(group_sizes)

# Surname and FamilySize
all_data['Surname'] = all_data['Name'].str.split().str[-1]
# Surname fill logic from notebook: ffill/bfill within Group
all_data['Surname'] = all_data.groupby('Group')['Surname'].ffill()
all_data['Surname'] = all_data.groupby('Group')['Surname'].bfill()
# Map FamilySize
family_counts = all_data['Surname'].value_counts()
all_data['FamilySize'] = all_data['Surname'].map(family_counts)
all_data.loc[all_data['Surname'].isna(), 'FamilySize'] = 1 # Fill unknown surname family size as 1

# HomePlanet Imputation
all_data['HomePlanet'] = all_data.groupby('Group')['HomePlanet'].ffill()
all_data['HomePlanet'] = all_data.groupby('Group')['HomePlanet'].bfill()
# HomePlanet from Surname
home_map = all_data.dropna(subset=['HomePlanet']).groupby('Surname')['HomePlanet'].agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
all_data['HomePlanet'] = all_data['HomePlanet'].fillna(all_data['Surname'].map(home_map))
all_data['HomePlanet'] = all_data['HomePlanet'].fillna(all_data['HomePlanet'].mode()[0])

# Cabin Imputation
# First split existing
# Wait, notebook split AFTER forward filling within group.
# But splitting creates nan columns if Cabin is nan.
# Notebook logic:
# 1. Split cabin (produces NaNs for missing Cabin) -> This was Cell 58.
# 2. Fill Deck/Side/Num within group.
# However, all_data['Cabin'] might still have NaNs.
# Let's perform split on what we have, then fill using group logic on the split columns.
all_data[['Deck', 'Num', 'Side']] = all_data['Cabin'].str.split('/', expand=True)

# Fill Deck/Side/Num within Group
for col in ['Deck', 'Side', 'Num']:
    all_data[col] = all_data.groupby('Group')[col].ffill()
    all_data[col] = all_data.groupby('Group')[col].bfill()

# If Num is filled, convert to int? might still have NaNs if group has no info.
# Notebook converted Num to numeric coercion.
# For remaining NaNs, notebook didn't explicitly say for global fill, 
# but AutoGluon handles NaNs. However, if we want to match exactly...
# Notebook Cell 75 output showed "0 missing". This implies Group fill was very effective or data is dense.
# Just in case, let's treat Num as numeric.

# Type Casting (Cell 81, 82)
all_data['CryoSleep'] = all_data['CryoSleep'].astype(int)
all_data['VIP'] = all_data['VIP'].astype(int)
# Num to int (handle NaNs if any remain)
all_data['Num'] = pd.to_numeric(all_data['Num'], errors='coerce')
# If NaNs remain in Num, fill with -1 or median? Notebook showed 0 missing.
if all_data['Num'].isna().sum() > 0:
    all_data['Num'] = all_data['Num'].fillna(all_data['Num'].median())
all_data['Num'] = all_data['Num'].astype(int)

# Drop columns not needed or text that AutoGluon might mishandle if not specified?
# AutoGluon handles text automatically.
# We should drop 'PassengerId' from features but keep for submission?
# AutoGluon ignores ID usually if specified.
# Let's drop 'Name' as we used Surname. 'Cabin' as we used components.
# 'Group' is ID-like.
cols_to_drop = ['Name', 'Cabin', 'Surname', 'Group'] 
# Note: Notebook didn't explicitly show dropping these before model, but typically we do.
# However, AutoGluon is robust. Let's drop 'Name' and 'Cabin' to be safe/clean.
all_data_final = all_data.drop(columns=['Name', 'Cabin'])

# 3. Split back
train_processed = all_data_final.iloc[:train_len].copy()
test_processed = all_data_final.iloc[train_len:].copy()

# Ensure target is present in train (Transported)
# And drop it from test (it should be NaN or we ignore it)
if 'Transported' in test_processed.columns:
    test_processed = test_processed.drop(columns=['Transported'])

print(f"Train shape: {train_processed.shape}")
print(f"Test shape: {test_processed.shape}")

# 4. Load Model and Predict
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model path {MODEL_PATH} not found.")
    exit(1)

print(f"Loading model from {MODEL_PATH}...")
predictor = TabularPredictor.load(MODEL_PATH)

print("Predicting...")
# AutoGluon can handle 'PassengerId' if passed, but usually we pass the dataframe.
# We'll rely on its automatic type inference.
y_pred = predictor.predict(test_processed)

# 5. Create Submission
print("Creating submission file...")
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Transported': y_pred
})

# Cast Transported to boolean if it's 0/1 or maintain string?
# Original target is boolean (True/False).
# Check prediction values.
if y_pred.dtype == 'int' or y_pred.dtype == 'int64':
    # If 0/1, map to True/False
    submission['Transported'] = submission['Transported'].astype(bool)

if not os.path.exists('submission'):
    os.makedirs('submission')

submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission saved to {SUBMISSION_PATH}")
print(submission.head())
