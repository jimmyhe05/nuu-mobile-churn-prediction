import pandas as pd
import numpy as np
import joblib
import os
import sys
import optuna
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import fbeta_score, classification_report, make_scorer
import logging
from sklearn.svm import SVC
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from sklearn.metrics import accuracy_score, precision_score, recall_score
import json

# =============================
# 🎯 Custom Fβ-score
# =============================
def custom_fbeta(y_true, y_pred, beta=2):
    return fbeta_score(y_true, y_pred, beta=beta)


user_beta = 2  # Change this if needed
metrics_file = "model_metrics.json"
fbeta_scorer = make_scorer(custom_fbeta, beta=user_beta)

# =============================
# 📂 Load & Preprocess Data (Dynamically from User Input)
# =============================
if len(sys.argv) < 2:
    print("❌ Usage: python train.py <path_to_processed_csv>")
    sys.exit(1)

file_path = sys.argv[1]

if not os.path.exists(file_path):
    print(f"❌ Error: File '{file_path}' not found!")
    sys.exit(1)

print(f"📂 Loading data from {file_path}...")
df_new = pd.read_csv(file_path)

# Ensure the required column exists
if 'churn' not in df_new.columns:
    print("❌ Error: 'churn' column not found in dataset. Please provide a valid dataset.")
    sys.exit(1)

# Drop any rows where 'churn' is NaN
df_new = df_new.dropna(subset=['churn'])

if df_new.empty:
    print("❌ No valid data to train on. Ensure churn values exist in dataset.")
    sys.exit(1)

X_new = df_new.drop(columns=['churn', 'num__churn'],
                    errors='ignore')  # ✅ Remove num__churn
y_new = df_new['churn']


# Ensure y_new has at least two classes (0 and 1)
if len(set(y_new)) < 2:
    print("❌ Training dataset contains only one class. Incremental training requires both churn and non-churn labels.")
    sys.exit(1)

# =============================
# ✅ Train-Test Split
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X_new, y_new, test_size=0.2, random_state=42, stratify=y_new
)

# undersampler = RandomUnderSampler(sampling_strategy=0.2, random_state=42)  # Make minority class 20% of the total
# X_train, y_train = undersampler.fit_resample(X_train, y_train)
# smote_tomek = SMOTETomek(sampling_strategy=0.2, random_state=42)  # Ensure minority class is 20% of total
# X_train, y_train = smote_tomek.fit_resample(X_train, y_train)

# Print class distributions
print(f"🟢 Training set: {y_train.value_counts(normalize=True)}")
print(f"🔴 Test set: {y_test.value_counts(normalize=True)}")

# =============================
# 📂 Check for Existing Model (Incremental Training)
# =============================
model_path = "best_churn_model.pkl"
incremental_training = False

if os.path.exists(model_path):
    print("🔄 Existing model found! Checking for incremental learning support...")
    try:
        # First, try to load as a Scikit-learn model
        model = joblib.load(model_path)
        model_type = type(model).__name__
        print(f"✅ Loaded existing model: {model_type}")

    except Exception as e:
        print(f"⚠️ Joblib loading failed: {e}")
        print("🔄 Trying to load as an XGBoost model...")

        try:
            # If joblib fails, assume it's an XGBoost model
            model = XGBClassifier()
            model.load_model(model_path)
            print("✅ Loaded existing XGBoost model.")
            model_type = "XGBoost"

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("⚠️ Model file may be corrupted. Retraining from scratch...")
            model = None
            model_type = None

    # Incremental training based on the model type
    if model_type == "XGBoost":
        print("✅ Performing incremental training for XGBoost...")
        model.fit(X_train, y_train, xgb_model=model_path)  # Incremental update
        model.save_model(model_path)  # Save updated model
        incremental_training = True

    elif hasattr(model, "partial_fit"):  # Supports incremental learning (SGDClassifier)
        print("✅ Performing incremental training for Scikit-learn model...")
        model.partial_fit(X_train, y_train, classes=np.array(
            [0, 1]))  # Incremental update
        joblib.dump(model, model_path)  # Save updated model
        incremental_training = True
    else:
        print("⚠️ Model does not support incremental learning. Retraining from scratch...")
        model = None

    # ✅ Evaluate model after incremental learning
    print("\n🔍 Evaluating Model After Incremental Update...")
    y_test_probs = model.predict_proba(X_test)[:, 1]
    y_test_pred_adjusted = (y_test_probs >= 0.2).astype(int)

    fbeta = custom_fbeta(y_test, y_test_pred_adjusted, beta=user_beta)
    print(
        f"✅ Incrementally Trained Model F{user_beta}-score on Test Set: {fbeta:.4f}")
    # ✅ Print classification report
    print(classification_report(y_test, y_test_pred_adjusted))

    metrics = {
        "precision": round(precision_score(y_test, y_test_pred_adjusted), 4),
        "recall": round(recall_score(y_test, y_test_pred_adjusted), 4),
        "accuracy": round(accuracy_score(y_test, y_test_pred_adjusted), 4),
        f"F{user_beta}_score": round(custom_fbeta(y_test, y_test_pred_adjusted, beta=user_beta), 4),
        "model_name": model_type
    }

    # Save updated metrics
    with open(metrics_file, "w") as f:
        json.dump(metrics, f)

    print(f"✅ Incremental training complete. Updated metrics: {metrics}")
    sys.exit(0)  # Exit after incremental training

    sys.exit(0)  # ✅ Exit after incremental training and evaluation

else:
    print("🚀 No existing model found. Training a new model from scratch...")
    model = None

# =============================
# ⚙️ Optuna Hyperparameter Tuning
# =============================


def tune_hyperparameters(model_name):
    """Optimizes hyperparameters for the given model using Optuna."""

    def objective(trial):
        """Defines the objective function for Optuna optimization."""

        if model_name == "SGD Logistic Regression":
            model = SGDClassifier(
                loss="log_loss",
                class_weight="balanced",
                random_state=42,
                max_iter=1000,
                alpha=trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
                penalty=trial.suggest_categorical(
                    "penalty", ["l2", "elasticnet"]),
            )

        elif model_name == "XGBoost":
            model = XGBClassifier(
                eval_metric="logloss",
                scale_pos_weight=5,
                random_state=42,
                n_estimators=trial.suggest_int("n_estimators", 100, 500),
                learning_rate=trial.suggest_float(
                    "learning_rate", 0.01, 0.2, log=True),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
            )

        elif model_name == "Support Vector Machine":
            model = SVC(
                C=trial.suggest_float("C", 0.01, 10.0, log=True),
                kernel=trial.suggest_categorical(
                    "kernel", ["poly", "rbf", "sigmoid"]),
                gamma=trial.suggest_categorical("gamma", ["scale", "auto"]),
                class_weight="balanced",
                probability=True,  # Enables predict_proba for thresholding
                random_state=42
            )

        # Train the model
        model.fit(X_train, y_train)
        y_pred_probs = model.predict_proba(X_test)[:, 1]
        y_pred_adjusted = (y_pred_probs >= 0.2).astype(int)

        return custom_fbeta(y_test, y_pred_adjusted, beta=user_beta)

    # Run Optuna optimization
    study = optuna.create_study(direction="maximize")

    try:
        study.optimize(objective, n_trials=10)
    except ValueError as e:
        print(f"⚠️ No trials completed: {e}")
        return {}

    return study.best_params if len(study.trials) > 0 else {}


# =============================
# 🏋️‍♂️ Train & Evaluate Models
# =============================
# =============================
# 🏋️‍♂️ Train & Evaluate Models
# =============================
if model is None or not incremental_training:
    models = {
        "SGD Logistic Regression": SGDClassifier(loss="log_loss", class_weight="balanced", random_state=42, max_iter=1000),
        "XGBoost": XGBClassifier(eval_metric='logloss', scale_pos_weight=5, random_state=42),
        "Support Vector Machine": SVC(probability=True, random_state=42)
    }

    best_fbeta_score = 0
    best_model_name = None
    best_model = None
    best_metrics = {}

    for name, model in models.items():
        print(f"\n🔄 Tuning hyperparameters for {name}...")
        best_params = tune_hyperparameters(name)

        print(f"\n🔄 Training {name} with best parameters: {best_params}...")
        model.set_params(**best_params)

        model.fit(X_train, y_train)
        y_test_probs = model.predict_proba(X_test)[:, 1]
        y_test_pred_adjusted = (y_test_probs >= 0.2).astype(int)

        precision = precision_score(y_test, y_test_pred_adjusted)
        recall = recall_score(y_test, y_test_pred_adjusted)
        accuracy = accuracy_score(y_test, y_test_pred_adjusted)
        fbeta = custom_fbeta(y_test, y_test_pred_adjusted, beta=user_beta)

        print(f"✅ {name} F{user_beta}-score on Test Set: {fbeta:.4f}")
        print(classification_report(y_test, y_test_pred_adjusted))

        # If this model is the best so far, update the best model and its metrics
        if fbeta > best_fbeta_score:
            best_fbeta_score = fbeta
            best_model = model
            best_model_name = name
            best_metrics = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "accuracy": round(accuracy, 4),
                f"F{user_beta}_score": round(fbeta, 4),
                "model_name": best_model_name
            }

    # =============================
    # 💾 Save Best Model & Metrics
    # =============================
    if best_model:
        if best_model_name == "XGBoost":
            best_model.save_model(model_path)  # Save XGBoost booster
        else:
            joblib.dump(best_model, model_path)  # Save other models

        # Save metrics only for the best model
        metrics_file = "model_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(best_metrics, f)

        print(f"\n🏆 Best Model: {best_model_name} with F{user_beta}-score: {best_fbeta_score:.4f}")
        print(f"✅ Best model and its metrics saved successfully!")
    else:
        print("❌ No valid model was found.")

