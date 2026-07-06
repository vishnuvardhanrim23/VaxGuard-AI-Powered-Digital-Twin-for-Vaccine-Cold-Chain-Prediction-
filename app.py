
# ============================================================
# VaxGuard: AI-Powered Digital Twin for Vaccine Cold Chain
# Failure Prediction and Recovery Recommendation
# ============================================================
# Run this file with:
# streamlit run app.py
# ============================================================

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import LabelEncoder


# ============================================================
# Page Config
# ============================================================

st.set_page_config(
    page_title="VaxGuard Dashboard",
    page_icon="🧊",
    layout="wide"
)


# ============================================================
# Styling
# ============================================================

st.markdown("""
<style>
.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #0B3D91;
}
.sub-title {
    font-size: 18px;
    color: #475569;
    margin-bottom: 20px;
}
.risk-low {
    background-color: #DCFCE7;
    color: #166534;
    padding: 8px 14px;
    border-radius: 10px;
    font-weight: 700;
}
.risk-medium {
    background-color: #FEF3C7;
    color: #92400E;
    padding: 8px 14px;
    border-radius: 10px;
    font-weight: 700;
}
.risk-high {
    background-color: #FEE2E2;
    color: #991B1B;
    padding: 8px 14px;
    border-radius: 10px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# Vaccine and Location Config
# ============================================================

VACCINE_CONFIG = {
    "BCG": {
        "min_temp": 2,
        "max_temp": 8,
        "dose_value": 35,
        "ideal_temp": 5
    },
    "OPV": {
        "min_temp": -20,
        "max_temp": -15,
        "dose_value": 22,
        "ideal_temp": -18
    },
    "Measles": {
        "min_temp": 2,
        "max_temp": 8,
        "dose_value": 45,
        "ideal_temp": 5
    },
    "Covishield": {
        "min_temp": 2,
        "max_temp": 8,
        "dose_value": 250,
        "ideal_temp": 5
    },
    "Covaxin": {
        "min_temp": 2,
        "max_temp": 8,
        "dose_value": 295,
        "ideal_temp": 5
    },
    "Pentavalent": {
        "min_temp": 2,
        "max_temp": 8,
        "dose_value": 65,
        "ideal_temp": 5
    },
    "Hepatitis B": {
        "min_temp": 2,
        "max_temp": 8,
        "dose_value": 55,
        "ideal_temp": 5
    }
}

CITY_COORDS = {
    "Bengaluru": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867),
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.7041, 77.1025),
    "Kolkata": (22.5726, 88.3639),
    "Pune": (18.5204, 73.8567),
    "Ahmedabad": (23.0225, 72.5714),
    "Kochi": (9.9312, 76.2673),
    "Jaipur": (26.9124, 75.7873),
}


# ============================================================
# Dataset Generation
# 200 units/records for each vaccine
# Total records = 7 vaccines x 200 = 1400
# ============================================================

@st.cache_data
def generate_dataset(seed=42):
    np.random.seed(seed)

    records = []
    records_per_vaccine = 200
    start_time = datetime.now() - timedelta(days=30)

    equipment_options = ["Normal", "Minor Issue", "Degraded", "Failure"]
    weather_options = ["Normal", "Hot", "Rain", "Extreme Heat"]
    cities = list(CITY_COORDS.keys())

    row_id = 1

    for vaccine_type, config in VACCINE_CONFIG.items():
        for _ in range(records_per_vaccine):

            city = np.random.choice(cities)
            lat, lon = CITY_COORDS[city]

            equipment_status = np.random.choice(
                equipment_options,
                p=[0.65, 0.20, 0.10, 0.05]
            )

            weather_condition = np.random.choice(
                weather_options,
                p=[0.55, 0.25, 0.12, 0.08]
            )

            ideal_temp = config["ideal_temp"]
            min_temp = config["min_temp"]
            max_temp = config["max_temp"]

            equipment_effect = {
                "Normal": 0,
                "Minor Issue": 1.2,
                "Degraded": 3.0,
                "Failure": 5.5
            }[equipment_status]

            weather_effect = {
                "Normal": 0,
                "Hot": 1.5,
                "Rain": 0.5,
                "Extreme Heat": 3.2
            }[weather_condition]

            temperature_c = ideal_temp + np.random.normal(0, 1.2) + equipment_effect + weather_effect

            if vaccine_type == "OPV":
                temperature_c = ideal_temp + np.random.normal(0, 2.0) + equipment_effect + weather_effect

            humidity_percent = np.clip(np.random.normal(62, 14), 30, 95)
            transportation_duration_hr = np.random.randint(2, 72)
            delay_hr = max(0, np.random.normal(3, 4))
            distance_km = np.random.randint(30, 1600)
            alarm_events = np.random.poisson(0.4)

            temp_deviation = 0
            if temperature_c < min_temp:
                temp_deviation = min_temp - temperature_c
            elif temperature_c > max_temp:
                temp_deviation = temperature_c - max_temp

            humidity_breach = 1 if humidity_percent > 80 else 0

            equipment_risk = {
                "Normal": 5,
                "Minor Issue": 18,
                "Degraded": 38,
                "Failure": 65
            }[equipment_status]

            weather_risk = {
                "Normal": 4,
                "Hot": 14,
                "Rain": 8,
                "Extreme Heat": 25
            }[weather_condition]

            risk_score = (
                8
                + temp_deviation * 11
                + humidity_breach * 10
                + transportation_duration_hr * 0.25
                + delay_hr * 1.5
                + alarm_events * 6
                + equipment_risk
                + weather_risk
            )

            risk_score = np.clip(risk_score + np.random.normal(0, 4), 0, 100)

            remaining_safe_time_hr = (
                72
                - temp_deviation * 8
                - transportation_duration_hr * 0.35
                - delay_hr * 1.1
                - alarm_events * 3
                - humidity_breach * 8
                - equipment_risk * 0.25
            )

            remaining_safe_time_hr = max(0, remaining_safe_time_hr)

            spoilage_percentage = np.clip(
                risk_score * np.random.uniform(0.45, 0.90),
                0,
                100
            )

            total_units = 200
            units_at_risk = int(total_units * spoilage_percentage / 100)
            financial_loss_inr = units_at_risk * config["dose_value"]

            cold_chain_failure = 1 if risk_score >= 60 else 0

            timestamp = start_time + timedelta(minutes=row_id * 30)

            records.append({
                "timestamp": timestamp,
                "shipment_id": f"SHP-{10000 + row_id}",
                "vaccine_type": vaccine_type,
                "city": city,
                "latitude": lat + np.random.normal(0, 0.06),
                "longitude": lon + np.random.normal(0, 0.06),
                "temperature_c": round(float(temperature_c), 2),
                "humidity_percent": round(float(humidity_percent), 2),
                "transportation_duration_hr": int(transportation_duration_hr),
                "delay_hr": round(float(delay_hr), 2),
                "distance_km": int(distance_km),
                "equipment_status": equipment_status,
                "weather_condition": weather_condition,
                "alarm_events": int(alarm_events),
                "total_units": total_units,
                "dose_value_inr": config["dose_value"],
                "risk_score_percent": round(float(risk_score), 2),
                "remaining_safe_time_hr": round(float(remaining_safe_time_hr), 2),
                "spoilage_percentage": round(float(spoilage_percentage), 2),
                "units_at_risk": units_at_risk,
                "financial_loss_inr": round(float(financial_loss_inr), 2),
                "cold_chain_failure": cold_chain_failure
            })

            row_id += 1

    return pd.DataFrame(records)


# ============================================================
# ML Training
# ============================================================

def prepare_ml_data(df):
    model_df = df.copy()

    encoders = {}
    categorical_cols = [
        "vaccine_type",
        "city",
        "equipment_status",
        "weather_condition"
    ]

    for col in categorical_cols:
        le = LabelEncoder()
        model_df[col] = le.fit_transform(model_df[col])
        encoders[col] = le

    features = [
        "vaccine_type",
        "city",
        "temperature_c",
        "humidity_percent",
        "transportation_duration_hr",
        "delay_hr",
        "distance_km",
        "equipment_status",
        "weather_condition",
        "alarm_events",
        "total_units",
        "dose_value_inr"
    ]

    return model_df, features, encoders


@st.cache_resource
def train_models(df, model_name):
    model_df, features, encoders = prepare_ml_data(df)

    X = model_df[features]
    y_class = model_df["cold_chain_failure"]
    y_safe_time = model_df["remaining_safe_time_hr"]
    y_loss = model_df["financial_loss_inr"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_class,
        test_size=0.2,
        random_state=42,
        stratify=y_class
    )

    X_train_reg, X_test_reg, y_safe_train, y_safe_test = train_test_split(
        X,
        y_safe_time,
        test_size=0.2,
        random_state=42
    )

    X_train_loss, X_test_loss, y_loss_train, y_loss_test = train_test_split(
        X,
        y_loss,
        test_size=0.2,
        random_state=42
    )

    if model_name == "Gradient Boosting":
        classifier = GradientBoostingClassifier(random_state=42)
        safe_time_model = GradientBoostingRegressor(random_state=42)
        loss_model = GradientBoostingRegressor(random_state=42)
    else:
        classifier = RandomForestClassifier(
            n_estimators=180,
            max_depth=12,
            random_state=42
        )
        safe_time_model = RandomForestRegressor(
            n_estimators=180,
            max_depth=12,
            random_state=42
        )
        loss_model = RandomForestRegressor(
            n_estimators=180,
            max_depth=12,
            random_state=42
        )

    classifier.fit(X_train, y_train)
    safe_time_model.fit(X_train_reg, y_safe_train)
    loss_model.fit(X_train_loss, y_loss_train)

    y_pred = classifier.predict(X_test)
    y_safe_pred = safe_time_model.predict(X_test_reg)
    y_loss_pred = loss_model.predict(X_test_loss)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "safe_time_mae": mean_absolute_error(y_safe_test, y_safe_pred),
        "safe_time_rmse": np.sqrt(mean_squared_error(y_safe_test, y_safe_pred)),
        "safe_time_r2": r2_score(y_safe_test, y_safe_pred),
        "loss_mae": mean_absolute_error(y_loss_test, y_loss_pred),
        "loss_rmse": np.sqrt(mean_squared_error(y_loss_test, y_loss_pred)),
        "loss_r2": r2_score(y_loss_test, y_loss_pred)
    }

    return {
        "classifier": classifier,
        "safe_time_model": safe_time_model,
        "loss_model": loss_model,
        "features": features,
        "encoders": encoders,
        "metrics": metrics
    }


# ============================================================
# Helper Functions
# ============================================================

def get_risk_category(score):
    if score < 35:
        return "Low Risk"
    elif score < 65:
        return "Medium Risk"
    return "High Risk"


def get_risk_color(score):
    if score < 35:
        return "risk-low"
    elif score < 65:
        return "risk-medium"
    return "risk-high"


def generate_recommendations(row):
    recommendations = []

    if row["risk_score_percent"] >= 75:
        recommendations.append("Critical risk detected. Transfer vaccines to nearest backup cold storage immediately.")
        recommendations.append("Dispatch emergency refrigerated vehicle or portable cold box.")
        recommendations.append("Notify supply chain manager and healthcare administrator.")

    elif row["risk_score_percent"] >= 60:
        recommendations.append("High risk detected. Increase monitoring frequency and inspect refrigeration equipment.")
        recommendations.append("Prepare alternate route or backup storage facility.")

    elif row["risk_score_percent"] >= 35:
        recommendations.append("Medium risk detected. Continue monitoring and check temperature stability.")
        recommendations.append("Verify packaging insulation and sensor readings.")

    else:
        recommendations.append("Low risk. Continue standard cold-chain monitoring.")

    if row["temperature_c"] > 8 and row["vaccine_type"] != "OPV":
        recommendations.append("Temperature breach: restore storage temperature to 2°C–8°C range.")

    if row["vaccine_type"] == "OPV" and row["temperature_c"] > -15:
        recommendations.append("OPV breach: transfer to deep-freeze storage immediately.")

    if row["humidity_percent"] > 80:
        recommendations.append("Humidity breach: inspect packaging, insulation, and condensation exposure.")

    if row["equipment_status"] in ["Degraded", "Failure"]:
        recommendations.append("Equipment issue: repair refrigeration unit or shift shipment to backup vehicle.")

    if row["remaining_safe_time_hr"] < 8:
        recommendations.append("Remaining safe time is very low. Prioritize emergency delivery.")

    return recommendations


def simulate_digital_twin(df, scenario, severity):
    sim_df = df.copy()

    if scenario == "Refrigeration Failure":
        sim_df["temperature_c"] += severity * 2.0
        sim_df["risk_score_percent"] += severity * 10
        sim_df["remaining_safe_time_hr"] -= severity * 6

    elif scenario == "Vehicle Breakdown":
        sim_df["delay_hr"] += severity * 3
        sim_df["risk_score_percent"] += severity * 8
        sim_df["remaining_safe_time_hr"] -= severity * 5

    elif scenario == "Traffic Delay":
        sim_df["delay_hr"] += severity * 2
        sim_df["risk_score_percent"] += severity * 6
        sim_df["remaining_safe_time_hr"] -= severity * 4

    elif scenario == "Temperature Spike":
        sim_df["temperature_c"] += severity * 2.5
        sim_df["risk_score_percent"] += severity * 9
        sim_df["remaining_safe_time_hr"] -= severity * 5

    elif scenario == "Power Outage":
        sim_df["temperature_c"] += severity * 2.2
        sim_df["risk_score_percent"] += severity * 11
        sim_df["remaining_safe_time_hr"] -= severity * 7

    elif scenario == "Unexpected Delay":
        sim_df["delay_hr"] += severity * 2.5
        sim_df["risk_score_percent"] += severity * 7
        sim_df["remaining_safe_time_hr"] -= severity * 4.5

    sim_df["risk_score_percent"] = np.clip(sim_df["risk_score_percent"], 0, 100)
    sim_df["remaining_safe_time_hr"] = np.clip(sim_df["remaining_safe_time_hr"], 0, None)
    sim_df["spoilage_percentage"] = np.clip(sim_df["risk_score_percent"] * 0.75, 0, 100)
    sim_df["units_at_risk"] = (sim_df["total_units"] * sim_df["spoilage_percentage"] / 100).astype(int)
    sim_df["financial_loss_inr"] = sim_df["units_at_risk"] * sim_df["dose_value_inr"]
    sim_df["cold_chain_failure"] = (sim_df["risk_score_percent"] >= 60).astype(int)

    return sim_df


def download_csv(df, filename, label):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv"
    )


# ============================================================
# Load Data and Models
# ============================================================

df = generate_dataset()

st.sidebar.title("🧊 VaxGuard")
model_choice = st.sidebar.selectbox(
    "Select ML Model",
    ["Random Forest", "Gradient Boosting"]
)

models = train_models(df, model_choice)

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Dashboard Pages",
    [
        "Executive Summary",
        "Active Shipment Monitoring",
        "Temperature & Humidity Trends",
        "Risk Prediction Dashboard",
        "Remaining Safe Time Prediction",
        "Vaccine Loss Analysis",
        "Financial Impact Analysis",
        "Geographical Shipment Tracking",
        "Recovery Recommendation Panel",
        "Digital Twin Simulation",
        "Model Performance Dashboard",
        "Dataset"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Dataset contains 200 units/records for each vaccine type. "
    "Total records: 1400."
)


# ============================================================
# Main Header
# ============================================================

st.markdown('<div class="main-title">VaxGuard: Vaccine Cold Chain Digital Twin Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Risk Prediction • Remaining Safe Time • Vaccine Loss • Financial Impact • Recovery Recommendation</div>', unsafe_allow_html=True)


# ============================================================
# Page 1: Executive Summary
# ============================================================

if page == "Executive Summary":

    total_shipments = len(df)
    high_risk_shipments = len(df[df["risk_score_percent"] >= 65])
    avg_risk = df["risk_score_percent"].mean()
    total_units = df["total_units"].sum()
    total_units_at_risk = df["units_at_risk"].sum()
    total_loss = df["financial_loss_inr"].sum()
    avg_safe_time = df["remaining_safe_time_hr"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Shipments", f"{total_shipments:,}")
    c2.metric("High-Risk Shipments", f"{high_risk_shipments:,}")
    c3.metric("Total Vaccine Units", f"{total_units:,}")
    c4.metric("Units at Risk", f"{total_units_at_risk:,}")

    c5, c6, c7 = st.columns(3)
    c5.metric("Average Risk Score", f"{avg_risk:.2f}%")
    c6.metric("Average Safe Time", f"{avg_safe_time:.2f} hr")
    c7.metric("Estimated Financial Loss", f"₹{total_loss:,.0f}")

    df_summary = df.copy()
    df_summary["risk_category"] = df_summary["risk_score_percent"].apply(get_risk_category)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(
            df_summary,
            names="risk_category",
            title="Risk Category Distribution",
            hole=0.45
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        vaccine_risk = df.groupby("vaccine_type")["risk_score_percent"].mean().reset_index()
        fig = px.bar(
            vaccine_risk,
            x="vaccine_type",
            y="risk_score_percent",
            title="Average Risk Score by Vaccine"
        )
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Page 2: Active Shipment Monitoring
# ============================================================

elif page == "Active Shipment Monitoring":

    st.subheader("Active Shipment Monitoring")

    selected_vaccines = st.multiselect(
        "Select Vaccine Type",
        sorted(df["vaccine_type"].unique()),
        default=sorted(df["vaccine_type"].unique())
    )

    selected_cities = st.multiselect(
        "Select City",
        sorted(df["city"].unique()),
        default=sorted(df["city"].unique())
    )

    filtered_df = df[
        (df["vaccine_type"].isin(selected_vaccines)) &
        (df["city"].isin(selected_cities))
    ]

    display_cols = [
        "timestamp",
        "shipment_id",
        "vaccine_type",
        "city",
        "temperature_c",
        "humidity_percent",
        "risk_score_percent",
        "remaining_safe_time_hr",
        "units_at_risk",
        "financial_loss_inr"
    ]

    st.dataframe(
        filtered_df[display_cols].sort_values("risk_score_percent", ascending=False),
        use_container_width=True
    )


# ============================================================
# Page 3: Temperature & Humidity Trends
# ============================================================

elif page == "Temperature & Humidity Trends":

    st.subheader("Temperature & Humidity Trends")

    vaccine_filter = st.selectbox(
        "Select Vaccine",
        ["All"] + sorted(df["vaccine_type"].unique().tolist())
    )

    trend_df = df.copy()

    if vaccine_filter != "All":
        trend_df = trend_df[trend_df["vaccine_type"] == vaccine_filter]

    trend_df = trend_df.sort_values("timestamp")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.line(
            trend_df,
            x="timestamp",
            y="temperature_c",
            color="vaccine_type",
            title="Temperature Trend"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.line(
            trend_df,
            x="timestamp",
            y="humidity_percent",
            color="vaccine_type",
            title="Humidity Trend"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Breach Analysis")

    breach_df = df.copy()
    breach_df["temperature_breach"] = breach_df.apply(
        lambda x: 1 if (
            x["temperature_c"] < VACCINE_CONFIG[x["vaccine_type"]]["min_temp"]
            or x["temperature_c"] > VACCINE_CONFIG[x["vaccine_type"]]["max_temp"]
        ) else 0,
        axis=1
    )
    breach_df["humidity_breach"] = (breach_df["humidity_percent"] > 80).astype(int)

    breach_summary = breach_df.groupby("vaccine_type")[["temperature_breach", "humidity_breach"]].sum().reset_index()

    fig = px.bar(
        breach_summary,
        x="vaccine_type",
        y=["temperature_breach", "humidity_breach"],
        barmode="group",
        title="Temperature and Humidity Breaches by Vaccine"
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Page 4: Risk Prediction Dashboard
# ============================================================

elif page == "Risk Prediction Dashboard":

    st.subheader("Cold Chain Failure Risk Prediction")

    c1, c2, c3 = st.columns(3)

    with c1:
        vaccine_type = st.selectbox("Vaccine Type", list(VACCINE_CONFIG.keys()))
        city = st.selectbox("City", list(CITY_COORDS.keys()))
        temperature_c = st.number_input("Temperature °C", value=5.0)

    with c2:
        humidity_percent = st.slider("Humidity %", 30, 100, 65)
        transportation_duration_hr = st.slider("Transportation Duration hr", 1, 100, 24)
        delay_hr = st.slider("Delay hr", 0, 48, 3)

    with c3:
        distance_km = st.slider("Distance km", 10, 2000, 500)
        equipment_status = st.selectbox("Equipment Status", ["Normal", "Minor Issue", "Degraded", "Failure"])
        weather_condition = st.selectbox("Weather Condition", ["Normal", "Hot", "Rain", "Extreme Heat"])
        alarm_events = st.slider("Alarm Events", 0, 10, 1)

    input_df = pd.DataFrame([{
        "vaccine_type": vaccine_type,
        "city": city,
        "temperature_c": temperature_c,
        "humidity_percent": humidity_percent,
        "transportation_duration_hr": transportation_duration_hr,
        "delay_hr": delay_hr,
        "distance_km": distance_km,
        "equipment_status": equipment_status,
        "weather_condition": weather_condition,
        "alarm_events": alarm_events,
        "total_units": 200,
        "dose_value_inr": VACCINE_CONFIG[vaccine_type]["dose_value"]
    }])

    encoded_input = input_df.copy()

    for col, encoder in models["encoders"].items():
        encoded_input[col] = encoder.transform(encoded_input[col])

    x_input = encoded_input[models["features"]]

    risk_probability = models["classifier"].predict_proba(x_input)[0][1] * 100
    risk_category = get_risk_category(risk_probability)

    safe_time_prediction = max(0, models["safe_time_model"].predict(x_input)[0])
    loss_prediction = max(0, models["loss_model"].predict(x_input)[0])
    spoilage_percentage = min(100, risk_probability * 0.75)
    units_at_risk = int(200 * spoilage_percentage / 100)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Risk Score", f"{risk_probability:.2f}%")
    m2.metric("Risk Category", risk_category)
    m3.metric("Predicted Units at Risk", f"{units_at_risk}")
    m4.metric("Predicted Financial Loss", f"₹{loss_prediction:,.0f}")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_probability,
        title={"text": "Cold Chain Failure Probability"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkblue"}
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Prediction Summary")
    st.write(f"Predicted remaining safe time: **{safe_time_prediction:.2f} hours**")
    st.write(f"Predicted spoilage percentage: **{spoilage_percentage:.2f}%**")


# ============================================================
# Page 5: Remaining Safe Time Prediction
# ============================================================

elif page == "Remaining Safe Time Prediction":

    st.subheader("Remaining Safe Time Prediction")

    safe_df = df.sort_values("remaining_safe_time_hr")

    c1, c2, c3 = st.columns(3)
    c1.metric("Minimum Safe Time", f"{safe_df['remaining_safe_time_hr'].min():.2f} hr")
    c2.metric("Average Safe Time", f"{safe_df['remaining_safe_time_hr'].mean():.2f} hr")
    c3.metric("Critical Shipments < 8 hr", f"{len(safe_df[safe_df['remaining_safe_time_hr'] < 8])}")

    fig = px.histogram(
        safe_df,
        x="remaining_safe_time_hr",
        nbins=30,
        title="Distribution of Remaining Safe Time"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Most Critical Shipments")
    st.dataframe(
        safe_df.head(25)[[
            "shipment_id",
            "vaccine_type",
            "city",
            "temperature_c",
            "humidity_percent",
            "risk_score_percent",
            "remaining_safe_time_hr",
            "units_at_risk",
            "financial_loss_inr"
        ]],
        use_container_width=True
    )


# ============================================================
# Page 6: Vaccine Loss Analysis
# ============================================================

elif page == "Vaccine Loss Analysis":

    st.subheader("Vaccine Loss Analysis")

    c1, c2 = st.columns(2)

    with c1:
        loss_by_vaccine = df.groupby("vaccine_type")["units_at_risk"].sum().reset_index()
        fig = px.bar(
            loss_by_vaccine,
            x="vaccine_type",
            y="units_at_risk",
            title="Total Units at Risk by Vaccine"
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        spoilage_by_vaccine = df.groupby("vaccine_type")["spoilage_percentage"].mean().reset_index()
        fig = px.bar(
            spoilage_by_vaccine,
            x="vaccine_type",
            y="spoilage_percentage",
            title="Average Spoilage Percentage by Vaccine"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Highest Vaccine Loss Shipments")
    st.dataframe(
        df.sort_values("units_at_risk", ascending=False).head(30)[[
            "shipment_id",
            "vaccine_type",
            "city",
            "total_units",
            "units_at_risk",
            "spoilage_percentage",
            "risk_score_percent",
            "remaining_safe_time_hr"
        ]],
        use_container_width=True
    )


# ============================================================
# Page 7: Financial Impact Analysis
# ============================================================

elif page == "Financial Impact Analysis":

    st.subheader("Financial Impact Analysis")

    total_loss = df["financial_loss_inr"].sum()
    average_loss = df["financial_loss_inr"].mean()
    max_loss = df["financial_loss_inr"].max()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Financial Loss", f"₹{total_loss:,.0f}")
    c2.metric("Average Loss per Shipment", f"₹{average_loss:,.0f}")
    c3.metric("Maximum Shipment Loss", f"₹{max_loss:,.0f}")

    col1, col2 = st.columns(2)

    with col1:
        loss_by_vaccine = df.groupby("vaccine_type")["financial_loss_inr"].sum().reset_index()
        fig = px.pie(
            loss_by_vaccine,
            names="vaccine_type",
            values="financial_loss_inr",
            title="Financial Loss Share by Vaccine"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        loss_by_city = df.groupby("city")["financial_loss_inr"].sum().reset_index()
        fig = px.bar(
            loss_by_city,
            x="city",
            y="financial_loss_inr",
            title="Financial Loss by City"
        )
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Page 8: Geographical Shipment Tracking
# ============================================================

elif page == "Geographical Shipment Tracking":

    st.subheader("Geographical Shipment Tracking")

    map_df = df.copy()
    map_df["risk_category"] = map_df["risk_score_percent"].apply(get_risk_category)

    fig = px.scatter_mapbox(
        map_df,
        lat="latitude",
        lon="longitude",
        color="risk_category",
        size="risk_score_percent",
        hover_name="shipment_id",
        hover_data=[
            "vaccine_type",
            "city",
            "temperature_c",
            "humidity_percent",
            "risk_score_percent",
            "remaining_safe_time_hr",
            "financial_loss_inr"
        ],
        zoom=4,
        height=650,
        title="Shipment Risk Map"
    )

    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Page 9: Recovery Recommendation Panel
# ============================================================

elif page == "Recovery Recommendation Panel":

    st.subheader("Recovery Recommendation Panel")

    critical_df = df.sort_values("risk_score_percent", ascending=False).head(20)

    for _, row in critical_df.iterrows():
        with st.expander(
            f"{row['shipment_id']} | {row['vaccine_type']} | {row['city']} | Risk: {row['risk_score_percent']}%"
        ):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Temperature", f"{row['temperature_c']} °C")
            c2.metric("Humidity", f"{row['humidity_percent']}%")
            c3.metric("Safe Time", f"{row['remaining_safe_time_hr']} hr")
            c4.metric("Financial Loss", f"₹{row['financial_loss_inr']:,.0f}")

            st.markdown("#### Recommended Recovery Actions")
            for rec in generate_recommendations(row):
                st.success("✅ " + rec)


# ============================================================
# Page 10: Digital Twin Simulation
# ============================================================

elif page == "Digital Twin Simulation":

    st.subheader("Digital Twin Simulation")

    scenario = st.selectbox(
        "Select Simulation Scenario",
        [
            "Refrigeration Failure",
            "Vehicle Breakdown",
            "Traffic Delay",
            "Temperature Spike",
            "Power Outage",
            "Unexpected Delay"
        ]
    )

    severity = st.slider("Scenario Severity", 1, 5, 3)
    sample_count = st.slider("Number of Shipments to Simulate", 50, 500, 200)

    base_df = df.sample(sample_count, random_state=10).copy()
    sim_df = simulate_digital_twin(base_df, scenario, severity)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Avg Risk Score",
        f"{sim_df['risk_score_percent'].mean():.2f}%",
        f"{sim_df['risk_score_percent'].mean() - base_df['risk_score_percent'].mean():.2f}%"
    )

    c2.metric(
        "Avg Safe Time",
        f"{sim_df['remaining_safe_time_hr'].mean():.2f} hr",
        f"{sim_df['remaining_safe_time_hr'].mean() - base_df['remaining_safe_time_hr'].mean():.2f} hr"
    )

    c3.metric(
        "Units at Risk",
        f"{sim_df['units_at_risk'].sum():,}",
        f"{sim_df['units_at_risk'].sum() - base_df['units_at_risk'].sum():,}"
    )

    c4.metric(
        "Financial Loss",
        f"₹{sim_df['financial_loss_inr'].sum():,.0f}",
        f"₹{sim_df['financial_loss_inr'].sum() - base_df['financial_loss_inr'].sum():,.0f}"
    )

    comparison = pd.DataFrame({
        "Metric": [
            "Average Risk Score",
            "Average Safe Time",
            "Average Spoilage %",
            "Average Financial Loss"
        ],
        "Before Simulation": [
            base_df["risk_score_percent"].mean(),
            base_df["remaining_safe_time_hr"].mean(),
            base_df["spoilage_percentage"].mean(),
            base_df["financial_loss_inr"].mean()
        ],
        "After Simulation": [
            sim_df["risk_score_percent"].mean(),
            sim_df["remaining_safe_time_hr"].mean(),
            sim_df["spoilage_percentage"].mean(),
            sim_df["financial_loss_inr"].mean()
        ]
    })

    fig = px.bar(
        comparison,
        x="Metric",
        y=["Before Simulation", "After Simulation"],
        barmode="group",
        title=f"Digital Twin Impact: {scenario}"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Simulated Shipment Data")
    st.dataframe(
        sim_df.sort_values("risk_score_percent", ascending=False),
        use_container_width=True
    )


# ============================================================
# Page 11: Model Performance Dashboard
# ============================================================

elif page == "Model Performance Dashboard":

    st.subheader("Model Performance Dashboard")

    metrics = models["metrics"]

    st.markdown("### Classification Model Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    c2.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
    c3.metric("Recall", f"{metrics['recall'] * 100:.2f}%")
    c4.metric("F1 Score", f"{metrics['f1'] * 100:.2f}%")

    st.markdown("### Remaining Safe Time Regression")
    r1, r2, r3 = st.columns(3)
    r1.metric("MAE", f"{metrics['safe_time_mae']:.2f} hr")
    r2.metric("RMSE", f"{metrics['safe_time_rmse']:.2f} hr")
    r3.metric("R² Score", f"{metrics['safe_time_r2']:.3f}")

    st.markdown("### Financial Loss Regression")
    l1, l2, l3 = st.columns(3)
    l1.metric("MAE", f"₹{metrics['loss_mae']:,.0f}")
    l2.metric("RMSE", f"₹{metrics['loss_rmse']:,.0f}")
    l3.metric("R² Score", f"{metrics['loss_r2']:.3f}")

    st.info(
        "These metrics are calculated on the generated project dataset. "
        "When real IoT cold-chain data is added, retrain the models and compare the performance."
    )


# ============================================================
# Page 12: Dataset
# ============================================================

elif page == "Dataset":

    st.subheader("Dataset")

    st.write("Each vaccine type has exactly 200 records/units.")
    vaccine_counts = df["vaccine_type"].value_counts().reset_index()
    vaccine_counts.columns = ["vaccine_type", "record_count"]

    st.dataframe(vaccine_counts, use_container_width=True)

    st.dataframe(df, use_container_width=True)

    download_csv(df, "vaxguard_dataset_200_per_vaccine.csv", "Download Dataset CSV")

    st.subheader("Dataset Summary")
    st.dataframe(df.describe(), use_container_width=True)
