import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta

# Title
st.title("Coal Price Forecasting App")
st.write("This app forecasts coal prices for the next 30 days based on external economic indicators.")

# Load trained model
model = joblib.load("xgb_model.joblib")

# Load base dataset (used during training)
df = pd.read_csv("merged_coal_externaldata.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Keep only one coal price column and relevant external factors
target_column = 'Coal Richards Bay 5500kcal NAR fob, London close, USD/t'
keep_cols = [
    'Date',
    target_column,
    'Crude Oil_Price', 'Brent Oil_Price', 'Dubai Crude_Price',
    'Natural Gas_Price',
    'Coal Richards Bay 4800kcal NAR fob, London close, USD/t',
    'Coal Richards Bay 5700kcal NAR fob, London close, USD/t',
    'Coal India 5500kcal NAR cfr, London close, USD/t',
    'Dutch TTF_Price'
]
df = df[keep_cols]

# Add lag features
def add_lag_features(df, lags=[1, 2, 3, 7, 14]):
    for lag in lags:
        df[f'lag_{lag}'] = df[target_column].shift(lag)
    return df

df = add_lag_features(df)
df.dropna(inplace=True)

# Prepare future forecast loop
future_dates = pd.date_range(start=df['Date'].max() + timedelta(days=1), periods=30)
last_known_price = df[target_column].iloc[-1]

external_cols = [
    target_column,
    'Crude Oil_Price', 'Brent Oil_Price', 'Dubai Crude_Price', 'Natural Gas_Price'
]

# Sidebar input
st.sidebar.header("External Factors Input")
user_inputs = {}
for col in external_cols:
    user_inputs[col] = st.sidebar.number_input(col, value=float(df[col].iloc[-1]))

# Initialize results storage
predictions = []
lag_values = {
    'lag_1': df[target_column].iloc[-1],
    'lag_2': df[target_column].iloc[-2],
    'lag_3': df[target_column].iloc[-3],
    'lag_7': df[target_column].iloc[-7],
    'lag_14': df[target_column].iloc[-14],
}

# Forecast loop with dynamic lag updates
for i in range(30):
    row = user_inputs.copy()
    row.update(lag_values)
    features_order = model.feature_names_in_

    # Fill in any missing features with recent values or 0
    missing_cols = [f for f in features_order if f not in row]
    if missing_cols:
        for col in missing_cols:
            if col in df.columns:
                row[col] = df[col].iloc[-1]  # Use last known value
            else:
                row[col] = 0  # Default to 0 if not available

    X_input = pd.DataFrame([row], columns=features_order)
    y_pred = model.predict(X_input)[0]
    predictions.append(y_pred)

    # Update lag values for next iteration
    for j in range(14, 0, -1):
        lag_values[f'lag_{j}'] = lag_values.get(f'lag_{j-1}', y_pred)
    lag_values['lag_1'] = y_pred

# Confidence interval (±2%)
lower_bound = [p * 0.98 for p in predictions]
upper_bound = [p * 1.02 for p in predictions]

# Display results
results = pd.DataFrame({
    "Date": future_dates,
    "Forecasted Price": predictions,
    "Lower Bound": lower_bound,
    "Upper Bound": upper_bound
})

st.subheader("📈 30-Day Forecast")
st.dataframe(results)
st.line_chart(results.set_index("Date"))
