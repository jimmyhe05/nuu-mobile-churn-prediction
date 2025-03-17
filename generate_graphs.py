from sklearn.metrics import roc_curve, auc
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import re
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt
import pandas as pd

# Load Processed Data
file_path = "processed_churn_data.xlsx"
df = pd.read_excel(file_path)

# Drop irrelevant columns
irrelevant_cols = ['device number', 'type', 'warranty', 'office date', 'office time in', 'warranty', 'defect / damage type', 'responsible party', 'final status', 'month', 'source', 'carrier']
df.drop(columns=[col for col in irrelevant_cols if col in df.columns], inplace=True)

# Identify Feature Columns
target_col = 'churn'
feature_cols = [col for col in df.columns if col != target_col]

# Ensure missing columns exist
required_columns = {'register_email', 'sim_info', 'promotion_email', 'days_since_activation', 'product/model #'}
for col in required_columns:
    if col not in df.columns:
        df[col] = np.nan  # Fill missing columns with NaN

# Convert Date Columns into Numeric Features
date_cols = ['active_date', 'last_boot_date', 'interval_date']
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        df[f'{col}_year'] = df[col].dt.year
        df[f'{col}_month'] = df[col].dt.month
        df[f'{col}_day'] = df[col].dt.day
        df.drop(columns=[col], inplace=True)

# Split Features and Target
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify categorical & numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Handle Missing Values
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

# Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[("imputer", num_imputer), ("scaler", StandardScaler())]), numerical_cols),
        ('cat', Pipeline(steps=[("imputer", cat_imputer), ("encoder", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))]), categorical_cols)
    ])

# Apply preprocessing
X_processed = preprocessor.fit_transform(X)

# Sanitize Feature Names
feature_names = preprocessor.get_feature_names_out()
feature_names = [re.sub(r'[^a-zA-Z0-9_]', '_', name) for name in feature_names]

# Convert processed data to DataFrame
X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

# Split Dataset (Use same test set)
_, X_test, _, y_test = train_test_split(X_processed_df, y, test_size=0.2, random_state=42, stratify=y)

# Load Best Model
best_model = joblib.load("best_churn_model.pkl")

# Get model name
model_name = best_model.named_steps['model'].__class__.__name__

# Get Probabilities for Churn
y_probs = best_model.predict_proba(X_test)[:, 1]

# Compute AUC-ROC
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f"{model_name} (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for random guessing
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - Best Model ({model_name})")
plt.legend()
plt.show()

# Extract the actual classifier from the pipeline
model = best_model.named_steps['model']  # Extract classifier from pipeline

# Check if the model has feature_importances_
if hasattr(model, 'feature_importances_'):
    feature_importances = model.feature_importances_
    
    # Convert to DataFrame and sort by importance
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Plot feature importance
    plt.figure(figsize=(12, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title(f'Feature Importance for {model_name}')
    plt.gca().invert_yaxis()  # Highest importance at the top
    plt.show()
else:
    print(f"The selected model ({model_name}) does not support feature importance visualization.")
