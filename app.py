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

# Prepare future forecast loop
future_dates = pd.date_range(start=df['Date'].max() + timedelta(days=1), periods=30)
last_known_price = df['Coal Richards Bay 5500kcal NAR fob, London close, USD/t'].iloc[-1]

external_cols = [
    'Coal Richards Bay 4800kcal NAR fob, London close, USD/t',
    'Coal Richards Bay 5500kcal NAR fob, London close, USD/t',
    'Coal Richards Bay 5700kcal NAR fob, London close, USD/t',
    'Coal India 5500kcal NAR cfr, London close, USD/t',
    'Crude Oil_Price', 'Brent Oil_Price', 'Dubai Crude_Price',
    'Dutch TTF_Price', 'Natural Gas_Price'
]

# Sidebar input
st.sidebar.header("External Factors Input")
user_inputs = {}
for col in external_cols:
    user_inputs[col] = st.sidebar.number_input(col, value=float(df[col].iloc[-1]))

# Initialize results storage
predictions = []
lag_values = {
    'lag_1': df['Coal Richards Bay 5500kcal NAR fob, London close, USD/t'].iloc[-1],
    'lag_2': df['Coal Richards Bay 5500kcal NAR fob, London close, USD/t'].iloc[-2],
    'lag_3': df['Coal Richards Bay 5500kcal NAR fob, London close, USD/t'].iloc[-3],
    'lag_7': df['Coal Richards Bay 5500kcal NAR fob, London close, USD/t'].iloc[-7],
    'lag_14': df['Coal Richards Bay 5500kcal NAR fob, London close, USD/t'].iloc[-14],
}

# Forecast loop with dynamic lag updates
for i in range(30):
    row = user_inputs.copy()
    row.update(lag_values)
    X_input = pd.DataFrame([row])
    y_pred = model.predict(X_input)[0]
    predictions.append(y_pred)

    # Update lag values for next iteration
    lag_values = {
        'lag_1': y_pred,
        'lag_2': lag_values['lag_1'],
        'lag_3': lag_values['lag_2'],
        'lag_7': lag_values['lag_6'] if 'lag_6' in lag_values else lag_values['lag_7'],
        'lag_14': lag_values['lag_13'] if 'lag_13' in lag_values else lag_values['lag_14'],
    }
    for j in range(14, 0, -1):
        lag_values[f'lag_{j}'] = lag_values.get(f'lag_{j-1}', y_pred)

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
