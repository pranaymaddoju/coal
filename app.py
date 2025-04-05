import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Coal Price Forecast", layout="wide")
st.title("🔮 30-Day Coal Price Forecasting App")

# Upload section
uploaded_file = st.file_uploader("📤 Upload your Excel file with coal prices", type=["xlsx"])

if uploaded_file:
    # Read file
    df = pd.read_excel(uploaded_file)

    # Required columns
    target_col = 'Coal Richards Bay 6000kcal NAR fob current week avg, No time stamp, USD/t'
    feature_cols = [
        'Coal Richards Bay 4800kcal NAR fob, London close, USD/t',
        'Coal Richards Bay 5500kcal NAR fob, London close, USD/t',
        'Coal Richards Bay 5700kcal NAR fob, London close, USD/t',
        'Coal India 5500kcal NAR cfr, London close, USD/t'
    ]

    try:
        df = df[['Date', target_col] + feature_cols].dropna()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        # Create lag features
        def create_lag_features(data, lags=10):
            df_lag = data.copy()
            for i in range(1, lags + 1):
                df_lag[f'lag_{i}'] = df_lag[target_col].shift(i)
            return df_lag.dropna()

        lags = 10
        df_lagged = create_lag_features(df, lags)

        X = df_lagged[[f'lag_{i}' for i in range(1, lags + 1)] + feature_cols]
        y = df_lagged[target_col]

        # Train model
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X, y)

        # Forecast next 30 days
        last_rows = df.iloc[-lags:].copy()
        future_preds = []
        future_dates = []

        for i in range(30):
            lag_values = last_rows[target_col].values[-lags:].tolist()
            ext_features = df[feature_cols].iloc[-1].values

            input_row = np.array(lag_values + list(ext_features)).reshape(1, -1)
            pred = model.predict(input_row)[0]

            future_preds.append(pred)
            date = df['Date'].iloc[-1] + pd.Timedelta(days=i+1)
            future_dates.append(date)

            new_row = pd.DataFrame([[date, pred] + list(ext_features)],
                                   columns=['Date', target_col] + feature_cols)
            last_rows = pd.concat([last_rows, new_row], ignore_index=True)

        forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted Price': future_preds})

        # Plotting
        st.subheader("📈 Forecast Plot")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df['Date'], df[target_col], label='Historical')
        ax.plot(forecast_df['Date'], forecast_df['Forecasted Price'], label='Forecast', linestyle='--')
        ax.set_title("Coal Price Forecast - Next 30 Days")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD/t)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Display forecast table
        st.subheader("📋 Forecast Table")
        st.dataframe(forecast_df)

        # Download button
        csv = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Forecast CSV", data=csv, file_name="coal_30_day_forecast.csv", mime='text/csv')

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
else:
    st.info("Please upload a valid Excel file with required columns.")
