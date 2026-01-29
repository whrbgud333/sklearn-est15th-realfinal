import os
import io
import base64
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import subprocess

app = FastAPI()

# Enable CORS for Flask frontend (running on port 5000)
origins = [
    "http://localhost:5000",
    "http://127.0.0.1:5000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Global variable to store the processing state/filename (simplified for single user demo)
CURRENT_FILE = None

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global CURRENT_FILE
    try:
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())
        CURRENT_FILE = file_location
        return {"message": f"File '{file.filename}' uploaded successfully", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analyze")
async def analyze_data():
    global CURRENT_FILE
    if not CURRENT_FILE:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    try:
        df = pd.read_csv(CURRENT_FILE)
        description = df.describe().to_json()
        head = df.head().to_json(orient="split")
        info = {
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "shape": df.shape,
            "missing_values": df.isnull().sum().to_dict()
        }
        return {"head": json.loads(head), "description": json.loads(description), "info": info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualize")
async def visualize_data():
    global CURRENT_FILE
    if not CURRENT_FILE:
        raise HTTPException(status_code=400, detail="No file uploaded")

    try:
        df = pd.read_csv(CURRENT_FILE)
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        
        plots = []
        
        # 1. Correlation Heatmap
        if not numeric_df.empty:
            plt.figure(figsize=(10, 8))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
            plt.title("Correlation Heatmap")
            
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode("utf-8")
            plots.append({"name": "Correlation Heatmap", "image": img_str})
            plt.close()

            # 2. Distribution of first few numeric columns
            for col in numeric_df.columns[:3]:
                plt.figure(figsize=(8, 6))
                sns.histplot(df[col].dropna(), kde=True)
                plt.title(f"Distribution of {col}")
                
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode("utf-8")
                plots.append({"name": f"Distribution of {col}", "image": img_str})
                plt.close()

        return {"plots": plots}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/preprocess")
async def preprocess_data():
    global CURRENT_FILE
    if not CURRENT_FILE:
        raise HTTPException(status_code=400, detail="No file uploaded")

    try:
        df = pd.read_csv(CURRENT_FILE)
        
        # Simple Preprocessing: 
        # 1. Fill missing numeric values with mean
        # 2. Fill missing categorical values with mode
        # 3. Drop duplicates
        
        initial_shape = df.shape
        
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())
            
        for col in categorical_cols:
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])
        
        df.drop_duplicates(inplace=True)
        
        # Save processed file
        processed_file_path = os.path.join(UPLOAD_DIR, "processed_" + os.path.basename(CURRENT_FILE))
        df.to_csv(processed_file_path, index=False)
        CURRENT_FILE = processed_file_path # Update current file to processed one
        
        return {
            "message": "Data preprocessed successfully",
            "initial_shape": initial_shape,
            "final_shape": df.shape,
            "processed_file": os.path.basename(processed_file_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model")
async def run_model(target_column: str = Form(...)):
    global CURRENT_FILE
    if not CURRENT_FILE:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    try:
        # Generate Python Code for Modeling
        code_content = f"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import json

# Load Data
data_path = r"{CURRENT_FILE}"
df = pd.read_csv(data_path)
target = "{target_column}"

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

results = {{}}

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
"""
        generated_script_path = os.path.join(UPLOAD_DIR, "generated_model.py")
        with open(generated_script_path, "w", encoding="utf-8") as f:
            f.write(code_content)
        
        # Execute the generated script
        result = subprocess.run(["python", generated_script_path], capture_output=True, text=True)
        
        if result.returncode != 0:
            return {"error": "Model execution failed", "stderr": result.stderr}
            
        model_results = json.loads(result.stdout)
        
        return {
            "message": "Model generated and trained successfully",
            "generated_code_path": generated_script_path,
            "results": model_results,
            "code_preview": code_content
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
