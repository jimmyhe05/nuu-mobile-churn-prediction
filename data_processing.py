import poplib
import pandas as pd
import sys
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib


DEBUG = False  # Set to True to save debug files

def preprocess_data(input_file, output_file, mode="train"):
    """
    Preprocesses the dataset, handling missing values, feature engineering, and normalizing columns.
    Mode:
        - "train": Filters out rows with missing churn values (only trains on labeled data).
        - "predict": Keeps all data (churn values are irrelevant).
    Saves the processed data as a new CSV file.
    """
    print(f"📂 Loading data from {input_file} in {mode} mode...")

    # Load data based on file type
    if input_file.endswith(".xls") or input_file.endswith(".xlsx"):
        xls = pd.ExcelFile(input_file, engine="xlrd")
        sheet_name = xls.sheet_names[0]  # Assume first sheet
        df = pd.read_excel(xls, sheet_name=sheet_name, engine="xlrd")
    else:
        df = pd.read_csv(input_file)
        
        # load metadata file
    with open("models_metadata.json", "r") as file:
      metadata = json.load(file)["models"]

    # ✅ Normalize column names
    df.columns = df.columns.str.lower().str.strip()

    # ✅ Store 'device_number' separately in a CSV for reattachment after prediction
    if "device number" in df.columns:
        df_device_numbers = df[["device number"]].copy()  # Keep only device numbers
        df.drop(columns=["device number"], inplace=True)

        # Save device numbers to a temporary file (can be deleted after use)
        df_device_numbers.to_csv("temp_device_numbers.csv", index=False)

    # ✅ Ensure 'active_date' exists
    if 'active_date' in df.columns:
        df['active_date'] = pd.to_datetime(df['active_date'], errors="coerce")
        if (mode == "train"):
            df = df.dropna(subset=['active_date'])
        else:
            df['active_date'].fillna(df['active_date'].median(), inplace=True)


    today = datetime.today()
    df['days_since_activation'] = (today - df['active_date']).dt.days

    # ✅ Ensure 'churn' column exists (needed for training)
    if 'churn' not in df.columns:
        df['churn'] = None  # Explicitly set as unknown

    # ✅ Keep existing churn values unchanged, only fill missing ones
    df.loc[df['churn'].isna() & (df['days_since_activation'] > 30), 'churn'] = 0  # Definitely NOT churned
    df['churn'] = df['churn'].astype('Int64')  # Keep NaNs as Int64 (if any)

    # ✅ Extract carrier information safely
    def extract_carrier(x):
        try:
            if isinstance(x, float) or x is None:  # Handle NaN or non-string values
                return None
            x = x.strip().lower()
            if x == "uninserted":
                return None
            data = json.loads(x)
            return data[0]['carrier_name'] if data else None
        except json.JSONDecodeError:
            return None

    if 'sim_info' in df.columns:
        df['carrier'] = df['sim_info'].apply(extract_carrier)
        df['sim_info'] = df['sim_info'].apply(lambda x: 0 if str(x).strip().lower() == "uninserted" else 1)
 
    # ✅ Convert date columns to datetime
    for col in ['interval_date', 'last_boot_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # ✅ Calculate days since last use
    df['most_recent_use'] = df[['interval_date', 'last_boot_date']].max(axis=1)
    df['days_since_last_use'] = (today - df['most_recent_use']).dt.days
    df['days_used_since_activation'] = (df['most_recent_use'] - df['active_date']).dt.days
    df.drop(columns=['most_recent_use'], inplace=True)

    print(df.columns)
    # generate dataset for each model
    for m in metadata:
        df_new = df
        model_name = m["model_name"]
        irrelevant_cols = m["irrelevant_columns"]

        df_new.drop(columns=[col for col in irrelevant_cols if col in df.columns], inplace=True)
    
        print(df.columns)

        if (model_name == "MLP"):
            # Identify Feature Columns
            target_col = 'churn'
            feature_cols = [col for col in df_new.columns if col != target_col]

            # Convert Date Columns into Numeric Features
            date_cols = ['active_date', 'last_boot_date', 'interval_date']
            for col in date_cols:
                if col in df_new.columns:
                    df_new[col] = pd.to_datetime(df_new[col], errors="coerce")
                    df_new[f'{col}_year'] = df_new[col].dt.year
                    df_new[f'{col}_month'] = df_new[col].dt.month
                    df_new[f'{col}_day'] = df_new[col].dt.day
                    df_new.drop(columns=[col], inplace=True)

            # Split Features and Target
            print(df_new.columns)
            X = df_new.drop(columns=[target_col])
            y = df_new[target_col]

            # Identify categorical & numerical columns after transformation
            categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
            numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

            # Handle Missing Values - Decided to fill them in. Perhaps just scrap instead tho.
            num_imputer = SimpleImputer(strategy="median")
            cat_imputer = SimpleImputer(strategy="most_frequent")

            # Preprocessing Pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', Pipeline(steps=[("imputer", num_imputer), ("scaler", StandardScaler())]), numerical_cols),
                    ('cat', Pipeline(steps=[("imputer", cat_imputer), ("encoder", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))]), categorical_cols)
                ]
            )
            # Fit preprocessor on data
            preprocessor.fit(X)

            # Save the fitted preprocessor
            joblib.dump(preprocessor, f"preprocessor_{model_name}.pkl" if model_name else "preprocessor.pkl")


            # Transform the data
            X_processed = preprocessor.transform(X)

            # Convert processed data back to a DataFrame
            X_processed_df_new = pd.DataFrame(X_processed, columns=preprocessor.get_feature_names_out())

            # Append target column back
            X_processed_df_new['churn'] = y.values

            # Save processed dataset
            X_processed_df_new.to_csv(f"processed_churn_data_{model_name}.csv" if model_name else "processed_churn_data.csv", index=False)

            print(f"{model_name}:" if model_name else "Original:")
            print("Data processing complete. Processed dataset and preprocessor saved.\n")
          
        else:
            # ✅ Ensure 'product/model #' exists, even if missing in the dataset
            df_new['product/model #'] = df_new.get('product/model #', "unknown").astype(str)

            # ✅ Convert Date Columns into Numeric Features
            date_cols = ['active_date', 'last_boot_date', 'interval_date']
            for col in date_cols:
                if col in df.columns:
                    df_new[col] = pd.to_datetime(df_new[col], errors="coerce")
                    df_new[f'{col}_year'] = df_new[col].dt.year
                    df_new[f'{col}_month'] = df_new[col].dt.month
                    df_new[f'{col}_day'] = df_new[col].dt.day
                    df_new.drop(columns=[col], inplace=True)
            
            # ✅ Ensure numerical columns exist
            required_numerical = [
                'sim_info', 'promotion_email', 'register_email',
                'days_since_activation', 'days_since_last_use', 'days_used_since_activation'
            ]
            for col in required_numerical:
                if col not in df_new.columns:
                    df_new[col] = 0  # Default to 0 if missing

            # ✅ Ensure categorical encoding for `product/model #`
            df_new['product_model_encoded'] = OrdinalEncoder().fit_transform(df_new[['product/model #']])
            df_new.drop(columns=['product/model #'], inplace=True)

            if mode == "predict":
            # Ensure only features that were present in training are used in prediction
                expected_features = ['sim_info', 'promotion_email', 'register_email',
                         'days_since_activation', 'days_since_last_use', 'days_used_since_activation',
                         'product_model_encoded',]
                df_new = df_new[[col for col in expected_features if col in df_new.columns]]

            # ✅ Identify categorical & numerical columns after transformation
            categorical_cols = df_new.select_dtypes(include=['object']).columns.tolist()
            numerical_cols = df_new.select_dtypes(include=['int64', 'float64']).columns.tolist()

            # ✅ Handle Missing Values
            num_imputer = SimpleImputer(strategy="median")
            cat_imputer = SimpleImputer(strategy="most_frequent")

            # ✅ Preprocessing Pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', Pipeline(steps=[("imputer", num_imputer), ("scaler", StandardScaler())]), numerical_cols),
                    ('cat', Pipeline(steps=[("imputer", cat_imputer), ("encoder", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))]), categorical_cols)
                ]
            )

            # ✅ Fit & Transform Data
            df_processed = preprocessor.fit_transform(df_new)

            # ✅ Convert processed data back to a DataFrame
            df_processed = pd.DataFrame(df_processed, columns=preprocessor.get_feature_names_out())

            # ✅ Append target column back (only for training mode)
            if mode == "train":
                df_processed['churn'] = df['churn'].values
                df_processed = df_processed.dropna(subset=['churn'])  # Remove unlabeled data

            # ✅ Save processed dataset
            df_processed.to_csv(output_file, index=False)

            # ✅ Debugging Output (Only if DEBUG=True)
            if DEBUG:
                print("✅ Final Processed Columns:", df_new.columns.tolist())  
                df_new.to_csv("debug_processed_data.csv", index=False)  

            print(f"✅ Data processing complete. Processed dataset saved to {output_file}")

# ✅ Run as script with arguments
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("❌ Usage: python data_processing.py <input_file> <output_file> <mode: train/predict>")
        sys.exit(1)

    input_path = sys.argv[1]  # File received from Flask API
    output_path = sys.argv[2]  # Processed file to be used in training or prediction
    mode = sys.argv[3].lower()  # Mode selection
    if mode not in ["train", "predict"]:
        print("❌ Error: Mode must be 'train' or 'predict'")
        sys.exit(1)

    preprocess_data(input_path, output_path, mode)
