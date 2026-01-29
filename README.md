# Titanic Survivor Prediction Service

This project provides a machine learning model to predict Titanic survival and a Gradio web interface.

## Prerequisites
Ensure the following Python packages are installed:
```bash
pip install pandas numpy scikit-learn gradio joblib
```

## Setup & Running
1.  **Train the Model**:
    Run the `myModel.ipynb` notebook or the helper script to train the model and generate `titanic_voting_model.pkl`.
    ```bash
    python train_and_save_model.py
    ```

2.  **Run the Web App**:
    Start the Gradio interface.
    ```bash
    python web_app.py
    ```
    The app will launch in your browser (usually at http://127.0.0.1:7860).

## Files
- `web_app.py`: The Gradio web application.
- `myModel.ipynb`: Jupyter notebook for data analysis and model creation.
- `train_and_save_model.py`: Python script equivalent of the notebook for quick training.
- `titanic_voting_model.pkl`: The trained model file (generated after training).
