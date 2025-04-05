import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
from datetime import timedelta

st.title("🔮 Coal Price Forecasting App")
st.write("This app forecasts coal prices for the next 30 days based on external economic factors.")

# Load trained model
model = joblib.load("xgb_model.joblib")

# Load base dataset (must be same dataset used during training)
df = pd.read_csv("merged_coal_externaldata.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Add lag features
def add_lag_features(df, lags=[1, 2, 3, 7, 14]):
    for lag in lags:
        df[f'lag_{lag}'] = df['coal_price'].shift(lag)
    return df

df = add_lag_features(df)
df.dropna(inplace=True)

# Get the latest row to generate future data
last_known = df.iloc[-1]
external_features = [
    'Coal Richards Bay 4800kcal NAR fob, London close, USD/t',
    'Coal Richards Bay 5500kcal NAR fob, London close, USD/t',
    'Coal Richards Bay 5700kcal NAR fob, London close, USD/t',
    'Coal India 5500kcal NAR cfr, London close, USD/t',
    'Crude Oil_Price',
    'Brent Oil_Price',
    'Dubai Crude_Price',
    'Dutch TTF_Price',
    'Natural Gas_Price'
]

lags = ['lag_1', 'lag_2', 'lag_3', 'lag_7', 'lag_14']
all_features = external_features + lags

future_data = []
current_date = last_known['Date']

# Generate 30-day forecasts
for i in range(30):
    row = last_known[external_features + lags].copy()
    row['Date'] = current_date + timedelta(days=1)
    pred_input = row[all_features].values.reshape(1, -1)
    prediction = model.predict(pred_input)[0]
    row['coal_price'] = prediction  # Needed for future lag values
    future_data.append(row)

    # Update lag values for next iteration
    for lag in lags:
        days_back = int(lag.split('_')[1])
        if i >= days_back:
            row[lag] = future_data[i - days_back]['coal_price']
        else:
            row[lag] = last_known['coal_price']  # fallback

    last_known = row.copy()
    current_date = row['Date']

# Create forecast dataframe
forecast_df = pd.DataFrame(future_data)
forecast_df = forecast_df[['Date', 'coal_price']].rename(columns={'coal_price': 'Forecasted Coal Price'})

# Show results
st.subheader("📆 30-Day Forecast")
st.dataframe(forecast_df)
st.line_chart(forecast_df.set_index('Date'))
