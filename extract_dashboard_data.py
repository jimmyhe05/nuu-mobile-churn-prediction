import pandas as pd
from datetime import datetime

import pandas as pd
from datetime import datetime


def calculate_app_usage_percentages(df, column_name):
    """Calculates the total usage percentage for each app based on all users."""
    app_usage = {}

    # Ensure column is treated as string
    df[column_name] = df[column_name].astype(str)

    # Iterate over each row in the dataset
    for entry in df[column_name].dropna():
        # Ensure the entry is a string
        if not isinstance(entry, str):
            continue

        # Split individual app usage entries
        apps = entry.split(", ")
        for app in apps:
            try:
                # Split into app name and time (e.g., "Wechat 3021s")
                name, time = app.rsplit(" ", 1)
                time = int(time.rstrip("s"))  # Convert time to integer
                if name in app_usage:
                    app_usage[name] += time
                else:
                    app_usage[name] = time
            except (ValueError, AttributeError):
                continue  # Skip any malformed data

    # Compute total usage time
    total_usage_time = sum(app_usage.values())

    # Calculate percentage for each app
    if total_usage_time > 0:
        app_usage_percentages = [
            {"app": app, "percentage": round(
                (time / total_usage_time) * 100, 2)}
            for app, time in sorted(app_usage.items(), key=lambda x: x[1], reverse=True)
        ]
    else:
        app_usage_percentages = []

    return app_usage_percentages


def extract_dashboard_data():
    """Extracts age range counts, activation counts, app usage percentages, and churn counts for API response."""
    # Load the Data File
    file_path = "UW_Churn_Pred_Data.xls"
    xls = pd.ExcelFile(file_path, engine="xlrd")

    # Load the sheets separately
    df_main = pd.read_excel(xls, sheet_name="Data", engine="xlrd")
    df_churn = pd.read_excel(
        xls, sheet_name="Data Before Feb 13", engine="xlrd")

    # ✅ Normalize column names
    df_main.columns = df_main.columns.str.lower().str.strip()
    df_churn.columns = df_churn.columns.str.lower().str.strip()

    # ✅ Process Churn Counts per Month (Only from df_churn)
    if 'office date' not in df_churn.columns or 'churn' not in df_churn.columns:
        raise KeyError(
            "Missing required columns ('office date', 'churn') in 'Data Before Feb 13' sheet.")

    # Convert 'office date' to datetime
    df_churn['office date'] = pd.to_datetime(
        df_churn['office date'], errors="coerce")

    # Ensure 'churn' is numeric and filter churn = 1
    df_churn['churn'] = pd.to_numeric(
        df_churn['churn'], errors="coerce").astype('Int64')

    # Churn Counts per Month
    df_churn['churn_month'] = df_churn['office date'].dt.to_period(
        "M").astype(str)
    churn_counts = [
        {"month": k, "churn_count": int(v)}
        for k, v in df_churn[df_churn['churn'] == 1]['churn_month'].value_counts().sort_index().items()
    ]

    # ✅ Process Age Range Counts (Only from df_main)
    age_range_counts = []
    if 'age range' in df_main.columns:
        age_range_counts = [
            {"range": k, "count": int(v)}
            for k, v in df_main['age range'].value_counts().sort_index().items()
        ]

    # ✅ Process Activation Counts (Only from df_main)
    if 'activate date' not in df_main.columns:
        raise KeyError("Column 'activate date' not found in 'Data' sheet.")

    df_main['activate date'] = pd.to_datetime(
        df_main['activate date'], errors="coerce")

    df_main['activation_month'] = df_main['activate date'].dt.to_period(
        "M").astype(str)
    activation_counts = [
        {"month": k, "count": int(v)}
        for k, v in df_main['activation_month'].value_counts().sort_index().items()
    ]

    # ✅ Process App Usage Percentages (Only from df_main)
    app_usage_percentages = []
    corrected_column_name = "app uage (s)"  # Ensure correct column name

    if corrected_column_name in df_main.columns:
        app_usage_percentages = calculate_app_usage_percentages(
            df_main, corrected_column_name)

    return {
        "age_range_counts": age_range_counts,
        "activation_counts": activation_counts,
        "app_usage_percentages": app_usage_percentages,
        "churn_counts_per_month": churn_counts
    }


# Example usage
if __name__ == "__main__":
    data = extract_dashboard_data()
    print(data)  # Print extracted data for testing
