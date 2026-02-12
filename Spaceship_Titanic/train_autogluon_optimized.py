import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
import os

# 1. Load Data
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
SUBMISSION_PATH = "submission/submission_optimized.csv"
MODEL_PATH = "AutogluonModels/ag-optimized-best-quality"

print("Loading data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# Combine for consistent preprocessing
train_len = len(train_df)
all_data = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)

print("Preprocessing data and Engineering Features...")

# --- Feature Engineering (Inherited from Spaceship_1.ipynb) ---

# 1. Spending Features
spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
all_data[spending_cols] = all_data[spending_cols].fillna(0)
all_data['TotalSpending'] = all_data[spending_cols].sum(axis=1)

# 2. CryoSleep Imputation (Logic: If spending > 0, CryoSleep is False. If 0, likely True)
# Note: Notebook logic might be slightly nuanced, but this is the core derivative.
all_data.loc[(all_data['CryoSleep'].isna()) & (all_data['TotalSpending'] > 0), 'CryoSleep'] = False
all_data.loc[(all_data['CryoSleep'].isna()) & (all_data['TotalSpending'] == 0), 'CryoSleep'] = True
all_data['CryoSleep'] = all_data['CryoSleep'].astype(bool)

# 3. Age & AgeGroup
all_data['Age'] = all_data['Age'].fillna(all_data['Age'].median())

def update_age_group(age):
    if age <= 4: return 'Baby'
    elif age <= 12: return 'Child'
    elif age <= 19: return 'Teenager'
    elif age <= 40: return 'Adult'
    elif age <= 60: return 'Middle Aged'
    else: return 'Senior'

all_data['AgeGroup'] = all_data['Age'].apply(update_age_group)

# 4. VIP Imputation
all_data.loc[(all_data['VIP'].isna()) & (all_data['TotalSpending'] == 0), 'VIP'] = False
all_data.loc[(all_data['VIP'].isna()) & (all_data['Age'] <= 19), 'VIP'] = False
all_data['VIP'] = all_data['VIP'].fillna(False).astype(bool)

# 5. Destination Imputation
dest_mode = all_data['Destination'].mode()[0]
all_data['Destination'] = all_data['Destination'].fillna(dest_mode)

# 6. Group & GroupSize (from PassengerId gggg_pp)
all_data['Group'] = all_data['PassengerId'].str.split('_').str[0]
group_sizes = all_data.groupby('Group').size()
all_data['GroupSize'] = all_data['Group'].map(group_sizes)

# 7. Surname & FamilySize
all_data['Surname'] = all_data['Name'].str.split().str[-1]
# Fill Surname within Group if possible
all_data['Surname'] = all_data.groupby('Group')['Surname'].ffill()
all_data['Surname'] = all_data.groupby('Group')['Surname'].bfill()
# Map FamilySize
family_counts = all_data['Surname'].value_counts()
all_data['FamilySize'] = all_data['Surname'].map(family_counts)
all_data.loc[all_data['Surname'].isna(), 'FamilySize'] = 1 # Default for unknown

# 8. HomePlanet Imputation
all_data['HomePlanet'] = all_data.groupby('Group')['HomePlanet'].ffill()
all_data['HomePlanet'] = all_data.groupby('Group')['HomePlanet'].bfill()
# Map based on Surname if still missing
home_map = all_data.dropna(subset=['HomePlanet']).groupby('Surname')['HomePlanet'].agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
all_data['HomePlanet'] = all_data['HomePlanet'].fillna(all_data['Surname'].map(home_map))
all_data['HomePlanet'] = all_data['HomePlanet'].fillna(all_data['HomePlanet'].mode()[0])

# 9. Cabin (Deck/Num/Side)
# Split existing
all_data[['Deck', 'Num', 'Side']] = all_data['Cabin'].str.split('/', expand=True)
# Fill within Group
for col in ['Deck', 'Num', 'Side']:
    all_data[col] = all_data.groupby('Group')[col].ffill()
    all_data[col] = all_data.groupby('Group')[col].bfill()

# Process Num (convert to int/float for model can digest numerical relationship if needed, though categorical often okay)
# Safest is to treat Num as numeric if possible, or object. Deck/Side are strictly object.
# AutoGluon handles both well. Let's ensure no NaNs remain if possible.
if all_data['Num'].isna().sum() > 0:
    all_data['Num'] = pd.to_numeric(all_data['Num'], errors='coerce')
    all_data['Num'] = all_data['Num'].fillna(all_data['Num'].median())

# --- Model Preparation ---

# Drop high cardinality/ID columns to prevent overfitting
cols_to_drop = ['PassengerId', 'Name', 'Cabin', 'Surname', 'Group']
# Note: 'Group' is ID-like but used for GroupSize. GroupSize is kept. Group ID itself is not useful for generalization.
# Keeping 'Surname' might overfit to specific families not in test? Yes, usage of FamilySize covers the feature value.
# So dropping Surname is correct.

train_final = all_data.iloc[:train_len].copy().drop(columns=cols_to_drop)
test_final = all_data.iloc[train_len:].copy().drop(columns=cols_to_drop)

# Ensure target is dropped from test if present (Transported)
if 'Transported' in test_final.columns:
    test_final = test_final.drop(columns=['Transported'])

# Train target
label = 'Transported'
train_final[label] = train_final[label].astype(bool) # Ensure boolean

print(f"Features used: {list(train_final.columns)}")
print("Starting AutoGluon Training...")

# AutoGluon Hyperparameters
# Using 'best_quality' preset which usually enables bagging and stacking.
# Explicitly setting num_bag_folds and num_stack_levels to ensure high performance as requested.
# Time limit set to 3600s (1 hour).

hyperparameters = {
    'CAT': {'depth': 6}, # Prevent overfitting in CatBoost by limiting depth slightly? Default is often 6-8.
    # We can leave others to default or specify empty to use defaults logic of 'best_quality'
}

predictor = TabularPredictor(
    label=label,
    eval_metric='accuracy',
    path=MODEL_PATH,
    problem_type='binary'
).fit(
    train_data=train_final,
    presets='best_quality',
    time_limit=3600, # 1 hour
    num_bag_folds=8, # High bagging
    num_stack_levels=2, # Stacking
    # hyperparameters=hyperparameters, # Uncomment to enforce constraints, but 'best_quality' explores well.
    # To reduce overfitting gap, using 'included_model_types' might be better IF we knew which overfit.
    # Usually GBMs are fine. DeepLearning might overfit on small tabular data.
    # exclusion logic:
    # excluded_model_types=['NN_TORCH', 'FASTAI'] 
)

print("Training Complete.")
print("Summary:")
print(predictor.fit_summary())

# --- Submission Generation ---

print("Generating predictions...")
y_pred = predictor.predict(test_final)

submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'], # Use original test_df to ensure correct ID matching
    'Transported': y_pred
})

if not os.path.exists('submission'):
    os.makedirs('submission')

submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission saved successfully to {SUBMISSION_PATH}")
