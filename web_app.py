import gradio as gr
import joblib
import pandas as pd
import numpy as np
import os

# Load model
model_path = "titanic_voting_model.pkl"
if not os.path.exists(model_path):
    print(f"Warning: {model_path} not found. Please ensure the model is trained.")
    model = None
else:
    model = joblib.load(model_path)

def predict_survival(pclass, sex, age, sibsp, parch, fare, embarked):
    if model is None:
        return "Model not loaded. Please train the model first."
    
    # Create DataFrame with correct column names and types
    try:
        data = pd.DataFrame({
            'Pclass': [int(pclass)],
            'Sex': [sex],
            'Age': [float(age)],
            'SibSp': [int(sibsp)],
            'Parch': [int(parch)],
            'Fare': [float(fare)],
            'Embarked': [embarked]
        })
        
        # Predict
        prediction = model.predict(data)[0]
        # Soft voting classifier supports predict_proba
        probs = model.predict_proba(data)[0]
        prob_survived = probs[1]
        
        if prediction == 1:
            return f"Survived (Probability: {prob_survived:.2%})"
        else:
            return f"Did Not Survive (Probability: {1-prob_survived:.2%})"
            
    except Exception as e:
        return f"Prediction Error: {str(e)}"

# Interface
demo = gr.Interface(
    fn=predict_survival,
    inputs=[
        gr.Dropdown(choices=[1, 2, 3], label="Pclass (Ticket Class)", value=3),
        gr.Radio(choices=['male', 'female'], label="Sex", value='male'),
        gr.Number(label="Age", value=25),
        gr.Number(label="SibSp (Siblings/Spouses)", value=0),
        gr.Number(label="Parch (Parents/Children)", value=0),
        gr.Number(label="Fare", value=7.25),
        gr.Radio(choices=['S', 'C', 'Q'], label="Embarked", value='S')
    ],
    outputs="text",
    title="Titanic Survivor Prediction Service",
    description="Predict whether a passenger would survive the Titanic disaster based on their details."
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
    #demo.launch(server_name="0.0.0.0", server_port=7860)


