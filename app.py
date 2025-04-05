import streamlit as st
import pandas as pd
import joblib

st.title("📊 Coal Price Forecasting (XGBoost Model)")

# Load the pre-trained model
model = joblib.load("xgb_coal_forecast_model.pkl")

# Load the dataset used for last known features
df = pd.read_csv("merged_coal_externaldata.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Identify your target column
target_column = 'Coal Richards Bay 6000kcal NAR fob current week avg, No time stamp, USD/t'

# Get last known features (excluding Date and target column)
features = df.drop(columns=['Date', target_column])
last_row = features.iloc[-1:]

# Create next 30 days
future_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=30)
future_features = pd.concat([last_row]*30, ignore_index=True)
future_features['Date'] = future_dates

# Predict
X_future = future_features.drop(columns=['Date'])
y_pred = model.predict(X_future)

# Prepare output
forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Forecasted Coal Price': y_pred
})

# Display
st.subheader("📆 Forecast for Next 30 Days")
st.dataframe(forecast_df)

# Plot
st.line_chart(forecast_df.set_index('Date'))

# Download option
csv = forecast_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Forecast", csv, "forecast_30_days.csv", "text/csv")
