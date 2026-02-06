from autogluon.tabular import TabularPredictor
import os

model_path = "AutogluonModels/ag-3600s-final"
if os.path.exists(model_path):
    try:
        predictor = TabularPredictor.load(model_path)
        print("Model loaded successfully.")
        # Sometimes feature_metadata_in is not directly eager loaded, ensuring access
        print("Features used:", predictor.feature_metadata_in.get_features())
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Model path {model_path} does not exist.")
