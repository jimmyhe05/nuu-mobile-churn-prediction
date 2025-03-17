from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging
import os
import subprocess
import xgboost as xgb
from flask_cors import CORS
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator
from extract_dashboard_data import extract_dashboard_data  # Import the function
import json


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Adjust for production security

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model storage
MODEL_PATH = "best_churn_model.pkl"

# =============================
# 🔄 Model Loading Function
# =============================
def load_model():
    """Loads the best trained model, detecting if it's XGBoost or Scikit-learn."""
    if not os.path.exists(MODEL_PATH):
        logger.warning("⚠ No trained model found. Please call /train_model before making predictions.")
        return None

    try:
        # Try loading as a Scikit-learn model
        model = joblib.load(MODEL_PATH)
        if isinstance(model, (BaseEstimator, XGBClassifier)):
            logger.info(f"✅ Scikit-learn model ({type(model).__name__}) loaded successfully.")
            return model

    except Exception as e:
        logger.warning(f"⚠ Joblib loading failed: {e}. Trying XGBoost format...")

    try:
        # If joblib fails, assume it's an XGBoost model
        model = XGBClassifier()
        model.load_model(MODEL_PATH)
        logger.info("✅ XGBoost model loaded successfully.")
        return model

    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return None

# Load the model if it exists, otherwise wait for training
model = load_model()

@app.route('/')
def home():
    return "Churn Prediction API is running!"

# =============================
# 🔄 Data Processing
# =============================
def process_file(input_path, output_path, mode):
    """
    Runs data_processing.py on the uploaded file to ensure proper formatting.
    - mode: "train" or "predict"
    """
    try:
        logger.info(f"🔄 Processing file: {input_path} in {mode} mode with data_processing.py...")

        # Run data_processing.py with the specified mode
        subprocess.run(["python", "data_processing.py", input_path, output_path, mode], check=True)

        logger.info("✅ File processed successfully.")
        return output_path  # Return processed file path

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error processing file with data_processing.py: {e}")
        raise ValueError("Failed to process file. Ensure data_processing.py runs correctly.")

def load_data(file, mode):
    """
    Loads CSV or Excel file into a pandas DataFrame after processing.
    - mode: "train" or "predict"
    """
    try:
        temp_input_path = f"temp_{file.filename}"
        temp_output_path = f"processed_{file.filename}"  # Different from processed_churn_data.csv

        file.save(temp_input_path)  # Save file to disk
        processed_file = process_file(temp_input_path, temp_output_path, mode)  # Process the file

        df = pd.read_csv(processed_file)  # Load the processed data
        os.remove(temp_input_path)  # Cleanup raw input file

        return df

    except Exception as e:
        raise ValueError(f"Error processing file: {str(e)}")

# =============================
# 🔮 Batch Prediction Endpoint
# =============================
@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Handles batch predictions after verifying model availability."""
    load_model()
    if model is None:
        return jsonify({"error": "⚠ No trained model found. Please call /train_model first."}), 500
    try:
        file = request.files['file']

        df = load_data(file, "predict")  # Process in "predict" mode
        df = df.drop(columns=['num__churn'], errors='ignore')
        
        # ✅ Load stored device numbers and reattach them before prediction
        if os.path.exists("temp_device_numbers.csv"):
            df_device_numbers = pd.read_csv("temp_device_numbers.csv")
            os.remove("temp_device_numbers.csv")  # Clean up after use
        else:
            df_device_numbers = None  # Handle case where the file is missing

        # Make predictions based on model type
        if isinstance(model, XGBClassifier):
            predictions_proba = model.predict_proba(df)[:, 1]  # XGBoost method
        else:
            predictions_proba = model.predict_proba(df)[:, 1]  # Scikit-learn method

        # Attach predictions
        df["churn_probability"] = predictions_proba.tolist()
        df["customer_number"] = range(1, len(df) + 1)
        
        # ✅ Reattach `device_number` to ensure rows match original input
        if df_device_numbers is not None:
            df = pd.concat([df_device_numbers, df], axis=1)  # Merge by row order


        return jsonify({"predictions": df.to_dict(orient='records')})

    except ValueError as e:
        logger.error(f"❌ {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"❌ Error processing batch prediction request: {e}")
        return jsonify({"error": str(e)}), 500

# =============================
# 🏋️ Model Training Endpoint
# =============================
@app.route('/train_model', methods=['POST'])
def train_model():
    """Handles model training (either full or incremental) by calling train.py."""
    global model

    try:
        file = request.files['file']
        df = load_data(file, "train")  # Process in "train" mode

        # Ensure 'churn' column exists
        if 'churn' not in df.columns:
            return jsonify({"error": "Missing 'churn' column in dataset."}), 400

        # Save processed dataset as input for training
        training_data_path = "processed_churn_data.csv"
        df.to_csv(training_data_path, index=False)

        # Call train.py for training
        logger.info("🚀 Starting training process...")
        subprocess.run(["python", "train.py", training_data_path], check=True)

        logger.info("✅ Model training complete. Reloading new model...")

        # Reload the trained model
        model = load_model()  

        # Read model metrics from JSON
        metrics_file = "model_metrics.json"
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
        else:
            metrics = {"error": "Metrics file not found."}

        return jsonify({
            "message": "✅ Model trained successfully and loaded.",
            "metrics": metrics
        }), 200

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error training model with train.py: {e}")
        return jsonify({"error": "Training failed. Check logs for details."}), 500
    except ValueError as e:
        logger.error(f"❌ {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"❌ Error retraining model: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/feature_importance', methods=['GET'])
def feature_importance():
    if model is None:
        return jsonify({"error": "No trained model found. Please call /train_model first."}), 500

    try:
        if isinstance(model, XGBClassifier):
            importance = model.feature_importances_
            feature_names = model.get_booster().feature_names 
        elif hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
            feature_names = getattr(model, "feature_names_in_", [f"feature_{i}" for i in range(len(importance))])
        else:
            return jsonify({"error": "❌ Model does not support feature importance extraction."}), 400

        # Sort features by importance and convert to Python floats
        importance_dict = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)

        return jsonify({
            "feature_importance": [{"feature": f, "importance": float(i)} for f, i in importance_dict]
        })

    except Exception as e:
        logger.error(f"❌ Error retrieving feature importance: {e}")
        return jsonify({"error": str(e)}), 500
    
# =============================
# 📊 New API Endpoint for Dashboard Data
# =============================
@app.route('/dashboard_data', methods=['GET'])
def get_dashboard_data():
    """Fetches dashboard metrics (churn per month, app usage, activation counts, etc.)"""
    try:
        data = extract_dashboard_data()  # Call your data extraction function
        return jsonify(data)
    except Exception as e:
        logger.error(f"❌ Error fetching dashboard data: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/check_model', methods=['GET'])
def check_model():
    """Checks if the best_churn_model.pkl file exists."""
    model_exists = os.path.exists(MODEL_PATH)
    return jsonify({"model_exists": model_exists})

if __name__ == '__main__':
    app.run(debug=True)
