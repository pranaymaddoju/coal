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

# Drop columns not used during training
unused_cols = ['Coal Richards Bay 6000kcal NAR fob current week avg, No time stamp, USD/t']
df.drop(columns=[col for col in unused_cols if col in df.columns], inplace=True)

# Add lag features
def add_lag_features(df, lags=[1, 2, 3, 7, 14]):
    target_column = 'Coal Richards Bay 5500kcal NAR fob, London close, USD/t'
    for lag in lags:
        df[f'lag_{lag}'] = df[target_column].shift(lag)
    return df

df = add_lag_features(df)
df.dropna(inplace=True)

# Prepare 30 future dates
future_dates = pd.date_range(start=df['Date'].max() + timedelta(days=1), periods=30)
last_row = df.iloc[-1:].copy()

external_cols = [
    'Coal Richards Bay 4800kcal NAR fob, London close, USD/t',
    'Coal Richards Bay 5500kcal NAR fob, London close, USD/t',
    'Coal Richards Bay 5700kcal NAR fob, London close, USD/t',
    'Coal India 5500kcal NAR cfr, London close, USD/t',
    'Crude Oil_Price', 'Brent Oil_Price', 'Dubai Crude_Price',
    'Dutch TTF_Price', 'Natural Gas_Price'
]

# Input from user for external features
st.sidebar.header("External Factors Input")
user_inputs = {}
for col in external_cols:
    user_inputs[col] = st.sidebar.number_input(col, value=float(last_row[col].values[0]))

# Create future DataFrame with user input and lag features
future_df = pd.DataFrame([user_inputs] * 30)
future_df['Date'] = future_dates
for lag in [1, 2, 3, 7, 14]:
    future_df[f'lag_{lag}'] = last_row[f'lag_{lag}'].values[0]

# Define feature columns
feature_cols = [
    'lag_1', 'lag_2', 'lag_3', 'lag_7', 'lag_14',
    'Coal Richards Bay 4800kcal NAR fob, London close, USD/t',
    'Coal Richards Bay 5500kcal NAR fob, London close, USD/t',
    'Coal Richards Bay 5700kcal NAR fob, London close, USD/t',
    'Coal India 5500kcal NAR cfr, London close, USD/t',
    'Crude Oil_Price', 'Brent Oil_Price', 'Dubai Crude_Price',
    'Dutch TTF_Price', 'Natural Gas_Price'
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
