import streamlit as st
import pandas as pd
import joblib

st.title("🧠 Coal Price Forecasting App (XGBoost)")
st.markdown("Predicts next 30 days of coal prices using external factors.")

# Load the model and dataset
model = joblib.load("xgb_coal_forecast_model.pkl")
df = pd.read_csv("merged_coal_externaldata.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Select your actual target column
target_col = 'Coal Richards Bay 6000kcal NAR fob current week avg, No time stamp, USD/t'

# Prepare last known features
features = df.drop(columns=['Date', target_col])
last_row = features.iloc[-1:]

# Create 30 future rows with same features
future_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=30)
future_features = pd.concat([last_row]*30, ignore_index=True)
future_features['Date'] = future_dates

# Predict
X_input = future_features.drop(columns=['Date'])
y_pred = model.predict(X_input)

# Create forecast dataframe
forecast = pd.DataFrame({
    'Date': future_dates,
    'Forecasted Coal Price': y_pred
})

# Display
st.subheader("📅 30-Day Coal Price Forecast")
st.dataframe(forecast)

# Plot
st.line_chart(forecast.set_index('Date'))

# Download button
csv = forecast.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Forecast CSV", csv, "forecast_30_days.csv", "text/csv")
