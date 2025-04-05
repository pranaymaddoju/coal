import streamlit as st
import pandas as pd
import joblib
import datetime

# Load model and training data
model = joblib.load("xgb_coal_forecast_model.pkl")
df = pd.read_csv("merged_coal_externaldata.csv")
df['Date'] = pd.to_datetime(df['Date'])

st.title("📊 Coal Price Forecasting (XGBoost)")

# Get last known feature row
features = df.drop(columns=['Date', 'Coal Richards Bay 6000kcal NAR fob current week avg, No time stamp, USD/t'])  # update with actual target column
last_row = features.iloc[-1:]

# Generate next 30 days
future_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=30)
future_features = pd.concat([last_row]*30, ignore_index=True)
future_features['Date'] = future_dates

# Predict
X_future = future_features.drop(columns=['Date'])
forecast = model.predict(X_future)

# Create forecast dataframe
forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Forecasted Coal Price': forecast
})

# Display
st.write("📆 Forecast for Next 30 Days:")
st.dataframe(forecast_df)

# Plot
st.line_chart(forecast_df.set_index('Date')['Forecasted Coal Price'])

# Download
csv = forecast_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Forecast", data=csv, file_name='xgb_future_forecast.csv')
