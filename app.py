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

# Add lag features
def add_lag_features(df, lags=[1, 2, 3, 7, 14]):
    target_column = 'Coal Richards Bay 5500kcal NAR fob, London close, USD/t'
    for lag in lags:
        df[f'lag_{lag}'] = df[target_column].shift(lag)
    return df

df = add_lag_features(df)
df.dropna(inplace=True)

# Get latest row and generate 30 future rows with same external features
last_row = df.iloc[-1:]
future_dates = pd.date_range(start=df['Date'].max() + timedelta(days=1), periods=30)

future_df = pd.concat([last_row]*30, ignore_index=True)
future_df['Date'] = future_dates

# Recompute lag columns on duplicated data
for lag in [1, 2, 3, 7, 14]:
    future_df[f'lag_{lag}'] = last_row[f'lag_{lag}'].values[0]

# Ensure only model-relevant features are used
feature_cols = [
    'Coal Richards Bay 4800kcal NAR fob, London close, USD/t',
    'Coal Richards Bay 5500kcal NAR fob, London close, USD/t',
    'Coal Richards Bay 5700kcal NAR fob, London close, USD/t',
    'Coal India 5500kcal NAR cfr, London close, USD/t',
    'Crude Oil_Price',
    'Brent Oil_Price',
    'Dubai Crude_Price',
    'Dutch TTF_Price',
    'Natural Gas_Price',
    'lag_1', 'lag_2', 'lag_3', 'lag_7', 'lag_14'
]

X_future = future_df[feature_cols]

# Predict future prices
y_pred = model.predict(X_future)

# Show results
results = pd.DataFrame({
    "Date": future_dates,
    "Forecasted Price": y_pred
})

st.subheader("📈 30-Day Forecast")
st.dataframe(results)
st.line_chart(results.set_index("Date"))
