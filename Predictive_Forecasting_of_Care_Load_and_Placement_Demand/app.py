import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="HHS Care Forecasting System",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# LOAD MODEL AND DATA
# ============================================================

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE_DIR, "gradient_boosting_model.pkl"))

@st.cache_data
def load_features():
    return joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))

@st.cache_data
def load_data():
    data = pd.read_csv(os.path.join(BASE_DIR, "ml_df.csv"))
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)
    return data

model = load_model()
feature_cols = load_features()
df = load_data()
# ============================================================
# TITLE
# ============================================================

st.title("📊 HHS Care Forecasting System")

st.markdown(
    """
    ### Machine Learning-Based Forecasting of Children in HHS Care

    This application uses a **Gradient Boosting Regressor** to analyze
    historical HHS care data and generate short-term forecasts.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Select a section:",
    [
        "Dashboard",
        "Historical Data",
        "Forecast",
        "Model Performance"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.header("📌 Dashboard")

    latest_date = df["Date"].max()
    latest_value = df.loc[df["Date"] == latest_date,
                          "Children in HHS Care"].iloc[0]

    previous_value = df["Children in HHS Care"].iloc[-2]

    change = latest_value - previous_value

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Latest HHS Care",
        f"{latest_value:,.0f}"
    )

    col2.metric(
        "Latest Date",
        latest_date.strftime("%d %b %Y")
    )

    col3.metric(
        "Change",
        f"{change:+,.0f}"
    )

    col4.metric(
        "Historical Records",
        f"{len(df):,}"
    )

    st.divider()

    st.subheader("HHS Care Trend")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        df["Date"],
        df["Children in HHS Care"],
        linewidth=2
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Children in HHS Care")
    ax.set_title("Historical HHS Care Trend")

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)


# ============================================================
# HISTORICAL DATA
# ============================================================

elif page == "Historical Data":

    st.header("📋 Historical Data")

    st.write(
        f"Dataset contains **{len(df)} observations** "
        f"from **{df['Date'].min().date()}** "
        f"to **{df['Date'].max().date()}**."
    )

    st.dataframe(
        df,
        use_container_width=True,
        height=500
    )

    st.subheader("Summary Statistics")

    st.dataframe(
        df["Children in HHS Care"].describe(),
        use_container_width=True
    )


# ============================================================
# FORECAST
# ============================================================

elif page == "Forecast":

    st.header("🔮 HHS Care Forecast")

    st.write(
        "Generate a short-term forecast using the trained "
        "Gradient Boosting model."
    )

    forecast_days = st.slider(
        "Number of forecast days",
        min_value=7,
        max_value=30,
        value=30
    )

    if st.button("Generate Forecast"):

        # Work on a copy
        history = df.copy()

        predictions = []

        for i in range(forecast_days):

            future_date = history["Date"].max() + pd.Timedelta(days=1)

            # ------------------------------------------------
            # Build lag features
            # ------------------------------------------------

            target = history["Children in HHS Care"]

            lag_1 = target.iloc[-1]
            lag_2 = target.iloc[-2]
            lag_3 = target.iloc[-3]

            lag_5 = target.iloc[-5]
            lag_10 = target.iloc[-10]

            rolling_mean_5 = target.tail(5).mean()
            rolling_std_5 = target.tail(5).std()

            # ------------------------------------------------
            # Transfer / discharge
            # ------------------------------------------------

            transfer = (
                history["Children transferred out of CBP custody"]
                if "Children transferred out of CBP custody" in history.columns
                else pd.Series([0])
            )

            discharge = (
                history["Children discharged from HHS Care"]
                if "Children discharged from HHS Care" in history.columns
                else pd.Series([0])
            )

            transfer_lag_1 = transfer.iloc[-1]
            transfer_lag_2 = transfer.iloc[-2]
            transfer_lag_3 = transfer.iloc[-3]
            transfer_lag_5 = transfer.iloc[-5]

            discharge_lag_1 = discharge.iloc[-1]
            discharge_lag_2 = discharge.iloc[-2]
            discharge_lag_3 = discharge.iloc[-3]
            discharge_lag_5 = discharge.iloc[-5]

            # ------------------------------------------------
            # Net pressure
            # ------------------------------------------------

            if "Net_Pressure" in history.columns:

                net_pressure_lag_1 = history[
                    "Net_Pressure"
                ].iloc[-1]

            else:

                net_pressure_lag_1 = (
                    transfer_lag_1 - discharge_lag_1
                )

            # ------------------------------------------------
            # Days since previous observation
            # ------------------------------------------------

            days_since_previous = (
                future_date - history["Date"].iloc[-1]
            ).days

            # ------------------------------------------------
            # Calendar features
            # ------------------------------------------------

            month = future_date.month
            year = future_date.year

            # ------------------------------------------------
            # Feature dictionary
            # ------------------------------------------------

            features = {

                "HHS_Care_Lag_1": lag_1,
                "HHS_Care_Lag_2": lag_2,
                "HHS_Care_Lag_3": lag_3,
                "HHS_Care_Lag_5": lag_5,
                "HHS_Care_Lag_10": lag_10,

                "HHS_Care_Rolling_Mean_5":
                    rolling_mean_5,

                "HHS_Care_Rolling_Std_5":
                    rolling_std_5,

                "Transfer_Lag_1":
                    transfer_lag_1,

                "Transfer_Lag_2":
                    transfer_lag_2,

                "Transfer_Lag_3":
                    transfer_lag_3,

                "Transfer_Lag_5":
                    transfer_lag_5,

                "Discharge_Lag_1":
                    discharge_lag_1,

                "Discharge_Lag_2":
                    discharge_lag_2,

                "Discharge_Lag_3":
                    discharge_lag_3,

                "Discharge_Lag_5":
                    discharge_lag_5,

                "Net_Pressure_Lag_1":
                    net_pressure_lag_1,

                "Days_Since_Previous":
                    days_since_previous,

                "Month":
                    month,

                "Year":
                    year
            }

            # ------------------------------------------------
            # Create model input in EXACT feature order
            # ------------------------------------------------

            X_future = pd.DataFrame(
                [[features.get(col, 0)
                  for col in feature_cols]],
                columns=feature_cols
            )

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            prediction = model.predict(X_future)[0]

            # Prevent impossible negative prediction
            prediction = max(0, prediction)

            predictions.append(prediction)

            # ------------------------------------------------
            # Add prediction to history for recursive forecasting
            # ------------------------------------------------

            new_row = {
                "Date": future_date,
                "Children in HHS Care": prediction
            }

            for col in history.columns:

                if col not in new_row:

                    if col in [
                        "Children transferred out of CBP custody",
                        "Children discharged from HHS Care",
                        "Net_Pressure"
                    ]:

                        new_row[col] = history[col].iloc[-1]

                    else:

                        new_row[col] = 0

            history = pd.concat(
                [
                    history,
                    pd.DataFrame([new_row])
                ],
                ignore_index=True
            )

        # ====================================================
        # DISPLAY FORECAST
        # ====================================================

        forecast_dates = pd.date_range(
            start=df["Date"].max() + pd.Timedelta(days=1),
            periods=forecast_days,
            freq="D"
        )

        forecast_df = pd.DataFrame({
            "Date": forecast_dates,
            "Forecast HHS Care": np.round(
                predictions, 2
            )
        })

        st.subheader(
            f"📅 {forecast_days}-Day Forecast"
        )

        st.dataframe(
            forecast_df,
            use_container_width=True
        )

        # ====================================================
        # GRAPH
        # ====================================================

        st.subheader("Historical + Forecast")

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(
            df["Date"],
            df["Children in HHS Care"],
            label="Historical"
        )

        ax.plot(
            forecast_df["Date"],
            forecast_df["Forecast HHS Care"],
            linestyle="--",
            label="Forecast"
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Children in HHS Care")
        ax.set_title(
            "HHS Care: Historical Data and Forecast"
        )

        ax.legend()

        plt.xticks(rotation=45)
        plt.tight_layout()

        st.pyplot(fig)

        # ====================================================
        # DOWNLOAD
        # ====================================================

        csv = forecast_df.to_csv(index=False)

        st.download_button(
            "⬇️ Download Forecast CSV",
            csv,
            "HHS_Care_Forecast.csv",
            "text/csv"
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.header("📈 Model Performance")

    results = pd.DataFrame({

        "Model": [
            "Naive",
            "ARIMA(0,1,1)",
            "ARIMA(1,1,1)",
            "ARIMA(1,1,0)",
            "Random Forest",
            "Gradient Boosting"
        ],

        "MAE": [
            186.506944,
            186.620030,
            186.639427,
            186.713023,
            70.015106,
            55.564308
        ],

        "RMSE": [
            227.228547,
            227.395029,
            227.423811,
            227.532804,
            89.091689,
            76.057461
        ]
    })

    st.dataframe(
        results,
        use_container_width=True
    )

    st.success(
        "Best performing model: Gradient Boosting"
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Gradient Boosting MAE",
        "55.56"
    )

    col2.metric(
        "Gradient Boosting RMSE",
        "76.06"
    )

    st.subheader("Feature Importance")

    importance = pd.DataFrame({

        "Feature": [
            "HHS_Care_Lag_1",
            "HHS_Care_Lag_2",
            "HHS_Care_Rolling_Mean_5",
            "HHS_Care_Lag_5",
            "HHS_Care_Lag_3",
            "HHS_Care_Lag_10",
            "Month",
            "Transfer_Lag_5",
            "Discharge_Lag_3",
            "Year"
        ],

        "Importance": [
            0.750371,
            0.119752,
            0.043647,
            0.031684,
            0.024518,
            0.022611,
            0.002024,
            0.001680,
            0.001229,
            0.000362
        ]
    })

    st.dataframe(
        importance,
        use_container_width=True
    )

    st.bar_chart(
        importance.set_index("Feature")
    )

    st.subheader("Interpretation")

    st.write(
        """
        The previous HHS Care observation is the most important
        predictor in the Gradient Boosting model. This indicates
        strong short-term temporal dependence in the HHS Care
        series. Recent lag values and the rolling mean also
        contribute significantly to the prediction.
        """
    )
