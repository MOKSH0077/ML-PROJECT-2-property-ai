
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ===============================
# Load Pickle Files (UNCHANGED)
# ===============================
price_model = joblib.load("price_model.pkl")
price_ct = joblib.load("price_column_transformer.pkl")
price_scaler = joblib.load("price_scaler.pkl")
price_feature_names = joblib.load("price_feature_names.pkl")

roi_model = joblib.load("roi_model.pkl")
roi_ct = joblib.load("roi_column_transformer.pkl")
roi_scaler = joblib.load("roi_scaler.pkl")
roi_feature_names = joblib.load("roi_feature_names.pkl")

# ===============================
# City & Tier (UNCHANGED)
# ===============================
tier_1 = ["Mumbai","Delhi","Bengaluru","Chennai","Hyderabad","Pune","Kolkata","Ahmedabad"]
tier_2 = ["Jaipur","Chandigarh","Indore","Bhopal","Nagpur","Lucknow","Kanpur","Patna",
          "Vadodara","Surat","Ghaziabad","Thane","Navi Mumbai","Gurgaon"]

cities = tier_1 + tier_2 + ["Ludhiana","Agra","Nashik","Faridabad","Meerut","Rajkot"]

def map_city_to_tier(city):
    if city in tier_1:
        return "Tier_1"
    elif city in tier_2:
        return "Tier_2"
    else:
        return "Tier_3"

# ===============================
# ROI Defaults (UNCHANGED)
# ===============================
city_avg_price_map = {c: 10000 for c in cities}
city_rental_yield_map = {c: 3.0 for c in cities}

for c in tier_1:
    city_avg_price_map[c] = 18000
    city_rental_yield_map[c] = 3.5

for c in tier_2:
    city_avg_price_map[c] = 12000
    city_rental_yield_map[c] = 3.0

# ===============================
# PREMIUM UI (UNCHANGED)
# ===============================
st.markdown("""
<style>
body {
    background: radial-gradient(circle at top, #0f172a, #020617);
}
.glass {
    background: rgba(15,23,42,0.92);
    backdrop-filter: blur(18px);
    border-radius: 22px;
    padding: 28px;
    border: 1px solid rgba(234,179,8,0.25);
    box-shadow: 0 0 45px rgba(234,179,8,0.15);
}
.section-title {
    font-size: 18px;
    color: #eab308;
    font-weight: 600;
    margin-bottom: 8px;
}
.metric-card {
    background: #020617;
    border-radius: 18px;
    padding: 22px;
    text-align: center;
    border: 1px solid rgba(16,185,129,0.35);
}
.metric-title {
    color: #94a3b8;
    font-size: 14px;
}
.metric-value {
    font-size: 32px;
    font-weight: 700;
    color: #eab308;
}
h1 {
    text-align: center;
    font-size: 44px;
    background: linear-gradient(90deg, #eab308, #10b981);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# TITLE
# ===============================
st.markdown("<h1> Cyber Estate </h1>", unsafe_allow_html=True)

# ===============================
# INPUT UI
# ===============================
st.markdown("<div class='glass'>", unsafe_allow_html=True)

st.markdown("<div class='section-title'>📍 Location & Amenities</div>", unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    city = st.selectbox("City", cities)

st.markdown("---")

st.markdown("<div class='section-title'>🏗️ Property Profile</div>", unsafe_allow_html=True)
c3, c4, c5 = st.columns(3)

with c3:
    property_type = st.selectbox(
        "Property Type",
        ["Apartment","Villa","Independent House","Plot"]
    )

is_plot = property_type == "Plot"

with c4:
    if is_plot:
        bhk = 0
        st.text_input(
            "Bedrooms (BHK)",
            value="Not applicable for Plot",
            disabled=True
        )
    else:
        bhk = st.slider("Bedrooms (BHK)", 1, 6, 2)

with c5:
    if is_plot:
        furnishing = "Unfurnished"
        st.selectbox(
            "Furnishing",
            ["Unfurnished"],
            disabled=True,
            help="Not applicable for plots"
        )
    else:
        furnishing = st.selectbox(
            "Furnishing",
            ["Unfurnished","Semi-Furnished","Fully-Furnished"]
        )

st.markdown("---")

with c2:
    if is_plot:
        parking = "None"
        st.selectbox(
            "Parking Type",
            ["None"],
            disabled=True,
            help="Parking not applicable for plots"
        )
    else:
        parking = st.selectbox("Parking Type", ["None","Open","Covered"])

st.markdown("---")

st.markdown("<div class='section-title'>📐 Space & Age</div>", unsafe_allow_html=True)
c6, c7 = st.columns(2)
with c6:
    builtup_area = st.slider("Built-up Area (sqft)", 200, 20000, 1200, step=50)
with c7:
    property_age = st.slider("Property Age (Years)", 0, 100, 5)

st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# PREDICTION (UNCHANGED)
# ===============================
if st.button("🚀 Evaluate Property", use_container_width=True):
    try:
        city_tier = map_city_to_tier(city)

        df_price = pd.DataFrame([{
            "City_Tier": city_tier,
            "Property_Type": property_type,
            "BHK": bhk,
            "Builtup_Area_sqft": builtup_area,
            "Property_Age_years": property_age,
            "Furnishing": furnishing,
            "Parking": parking
        }])

        df_roi = df_price.copy()
        df_roi["Average_Area_Price_sqft"] = city_avg_price_map[city]
        df_roi["Rental_Yield_percent"] = city_rental_yield_map[city]

        X_price = pd.DataFrame(price_ct.transform(df_price), columns=price_feature_names)
        X_price.iloc[:, [18,19]] = price_scaler.transform(X_price.iloc[:, [18,19]])
        price = max(np.expm1(price_model.predict(X_price)[0]), 300000)

        X_roi = pd.DataFrame(roi_ct.transform(df_roi), columns=roi_feature_names)
        X_roi.iloc[:, [18,19,20,21]] = roi_scaler.transform(
            X_roi.iloc[:, [18,19,20,21]]
        )
        roi_pred = int(roi_model.predict(X_roi)[0])
        roi_label = "High ROI" if roi_pred == 1 else "Low–Medium ROI"

        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Estimated Property Value</div>
                <div class="metric-value">₹ {price/100000:.2f} Lakhs</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Investment Outlook</div>
                <div class="metric-value">{roi_label}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### 📊 Valuation Analytics")
        low, mid, high = price*0.85/100000, price/100000, price*1.15/100000

        g1, g2 = st.columns(2)
        with g1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=mid,
                title={"text":"Property Price (Lakhs)"},
                gauge={"axis":{"range":[0, high*1.3]},
                       "bar":{"color":"#eab308"}}
            ))
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)

        with g2:
            fig2 = go.Figure(go.Bar(
                x=["Low","Expected","High"],
                y=[low,mid,high],
                marker_color=["#64748b","#eab308","#10b981"]
            ))
            fig2.update_layout(title="Price Confidence Range (Lakhs)", height=320)
            st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error: {e}")
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
#
