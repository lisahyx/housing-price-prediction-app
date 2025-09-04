import streamlit as st
import shap
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib

# --- Page setup ---
st.set_page_config(
    page_title="Property Price Predictor",
    page_icon="🏠",
    layout="wide",
)

# --- Load model ---
@st.cache_resource
def load_model():
    try:
        model = xgb.Booster()
        model.load_model("xgboost_model.json")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def forward_size_transform(raw_size):
    """Apply same transform as training"""
    log_size = np.log1p(raw_size)
    norm_size = size_scaler.transform([[log_size]])[0][0]
    return norm_size
    
# Reverse transformation functions
def reverse_log1p_transform(log_value):
    """Reverse log1p transformation: exp(value) - 1"""
    return np.exp(log_value) - 1
    
model = load_model()

# Load scalers
size_scaler = joblib.load("size_scaler.pkl")
price_scaler = joblib.load("price_scaler.pkl")
feature_scaler = joblib.load("feature_scaler.pkl")

# --- Title & description ---
st.title("🏠 Property Price Prediction with SHAP Explainability")
st.markdown("""
This app predicts the property price based on various features and explains the prediction using SHAP values.  
Adjust the inputs below and see how they affect the prediction in real-time.
""")

# --- Features list (from your training pipeline) ---
# Extract location options from feature names (remove 'location_' prefix) 
all_features = ['size', 'bedrooms', 'bathrooms', 'security', 'parking', 'swimming_pool', 'gym', 'playground', 'sauna', 'tennis_court', 'bbq_area', 'clubhouse', 'jogging_track', 'multipurpose_hall', 'badminton_court', 'basketball_court', 'jacuzzi', 'business_center', 'cafe', 'community_garden', 'laundry', 'lounge', 'retail_stores', 'nursery', 'surau', 'solar_panels', 'pet_friendly', 'recreation_lake', 'recreation_room', 'lift', 'covered_linkways', 'bus_stop', 'drop_off_point', 'location_Alor Setar', 'location_Ambang Botanic', 'location_Ampang', 'location_Ampang Hilir', 'location_Ampang Jaya', 'location_Ara Damansara', 'location_Bachok', 'location_Balakong', 'location_Bandar Baru Bangi', 'location_Bandar Baru Cheras', 'location_Bandar Baru Klang', 'location_Bandar Baru Klang 2', 'location_Bandar Baru Permas Jaya', 'location_Bandar Baru Selayang', 'location_Bandar Baru Sentul', 'location_Bandar Baru Seri Kembangan', 'location_Bandar Baru UDA', 'location_Bandar Botanic', 'location_Bandar Bukit Puchong', 'location_Bandar Bukit Raja', 'location_Bandar Bukit Tinggi', 'location_Bandar Damai Perdana', 'location_Bandar Kinrara', 'location_Bandar Mahkota Cheras', 'location_Bandar Menjalara', 'location_Bandar Puchong', 'location_Bandar Puchong Jaya', 'location_Bandar Puchong Jaya Barat', 'location_Bandar Puchong Jaya Timur', 'location_Bandar Puchong Perdana', 'location_Bandar Puchong Utama 2', 'location_Bandar Puchong Utama 3', 'location_Bandar Puchong Utama 5', 'location_Bandar Puchong Utama Barat 2', 'location_Bandar Puteri Bangi', 'location_Bandar Puteri Kajang', 'location_Bandar Puteri Klang', 'location_Bandar Puteri Klang 2', 'location_Bandar Puteri Puchong', 'location_Bandar Puteri Puchong Barat', 'location_Bandar Putra Permai', 'location_Bandar Saujana Putra', 'location_Bandar Saujana Putra 2', 'location_Bandar Seri Putra', 'location_Bandar Sri Damansara', 'location_Bandar Sri Permaisuri', 'location_Bandar Sungai Long', 'location_Bandar Sunway', 'location_Bandar Tasik Permai', 'location_Bandar Tasik Permaisuri', 'location_Bandar Tasik Puteri', 'location_Bandar Tasik Selatan', 'location_Bandar Tun Razak', 'location_Bandar Utama', 'location_Bandaraya Melaka', 'location_Bangi', 'location_Bangsar', 'location_Bangsar South', 'location_Banting', 'location_Batang Kali', 'location_Batu', 'location_Batu 9 Cheras', 'location_Batu Caves', 'location_Batu Pahat', 'location_Bentong', 'location_Beranang', 'location_Besut', 'location_Bintulu', 'location_Brickfields', 'location_Bukit Beruntung', 'location_Bukit Bintang', 'location_Bukit Jalil', 'location_Bukit Jelutong', 'location_Bukit Kiara', 'location_Bukit Rahman Putra', 'location_Bukit Raja', 'location_Bukit Rimau', 'location_Bukit Tunku', 'location_Bukit Tunku (Kenny Hills)', 'location_Cameron Highlands', 'location_Chan Sow Lin', 'location_Cheras', 'location_Cheras (Kuala Lumpur)', 'location_Cheras Batu 9', 'location_Cheras Hartamas', 'location_City Centre', 'location_Country Heights', 'location_Country Heights Damansara', 'location_Cyberjaya', 'location_Damansara', 'location_Damansara Damai', 'location_Damansara Heights', 'location_Damansara Jaya', 'location_Damansara Perdana', 'location_Damansara Utama', 'location_Danau Kota', 'location_Denai Alam', 'location_Dengkil', 'location_Desa Pandan', 'location_Desa ParkCity', 'location_Desa Parkcity', 'location_Desa Petaling', 'location_Dungun', 'location_Dutamas', 'location_Glenmarie', 'location_Gombak', 'location_Gua Musang', 'location_Hulu Langat', 'location_Hulu Terengganu', 'location_I-City', 'location_Ijok', 'location_Ipoh', 'location_Jalan Duta', 'location_Jalan Ipoh', 'location_Jalan Klang Lama', 'location_Jalan Klang Lama (Old Klang Road)', 'location_Jalan Kuching', 'location_Jalan Sultan Ismail', 'location_Jeli', 'location_Jerantut', 'location_Jinjang', 'location_Johor Bahru', 'location_KL City', 'location_KL City Centre', 'location_KL Eco City', 'location_KLCC', 'location_Kajang', 'location_Kajang Perdana', 'location_Kajang Prima', 'location_Kajang Sentral', 'location_Kampung Kerinchi (Bangsar South)', 'location_Kangar', 'location_Kapar', 'location_Kayu Ara', 'location_Kelana Jaya', 'location_Kemaman', 'location_Keningau', 'location_Kenny Hills', 'location_Kepong', 'location_Kepong Baru', 'location_Kepong Entrepreneurs Park', 'location_Keramat', 'location_Kl Sentral', 'location_Klang', 'location_Kota Damansara', 'location_Kota Kemuning', 'location_Kota Kinabalu', 'location_Kota Marudu', 'location_Kuala Krai', 'location_Kuala Langat', 'location_Kuala Lumpur', 'location_Kuala Nerus', 'location_Kuala Penyu', 'location_Kuala Selangor', 'location_Kuala Terengganu', 'location_Kuantan', 'location_Kuchai Lama', 'location_Kuching', 'location_Kundang', 'location_Labuan', 'location_Lahad Datu', 'location_Marang', 'location_Melaka', 'location_Mid Valley City', 'location_Miri', 'location_Mont Kiara', 'location_Mutiara Damansara', 'location_OU G', 'location_OUG', 'location_Old Klang Road', 'location_Others', 'location_Pandamaran', 'location_Pandan Indah', 'location_Pandan Jaya', 'location_Pandan Perdana', 'location_Pangsapuri Cheras Perdana', 'location_Pantai', 'location_Pasar Seni', 'location_Pasir Puteh', 'location_Pelabuhan Klang', 'location_Penang', 'location_Petaling Jaya', 'location_Port Klang', 'location_Puchong', 'location_Puchong Bandar Puteri', 'location_Puchong Hartamas', 'location_Puchong Indah', 'location_Puchong Jaya', 'location_Puchong Kinrara', 'location_Puchong Perdana', 'location_Puchong Permai', 'location_Puchong Prima', 'location_Puchong South', 'location_Puchong Tenaga', 'location_Puchong Utama', 'location_Pudu', 'location_Puncak Alam', 'location_Pusat Bandar Puchong', 'location_Putra Heights', 'location_Raub', 'location_Rawang', 'location_Rawang Perdana', 'location_Salak Selatan', 'location_Salak Tinggi', 'location_Sandakan', 'location_Saujana', 'location_Saujana Utama', 'location_Segambut', 'location_Seksyen 13 Shah Alam', 'location_Seksyen 17 Shah Alam', 'location_Seksyen 7 Shah Alam', 'location_Selangor', 'location_Selayang', 'location_Selayang Baru', 'location_Semenyih', 'location_Sentul', 'location_Sepang', 'location_Seputeh', 'location_Serdang', 'location_Seremban', 'location_Serendah', 'location_Seri Kembangan', 'location_Setapak', 'location_Setapak Permai', 'location_Setia Alam', 'location_Setia Eco Park', 'location_Setia Indah', 'location_Setiawangsa', 'location_Setiu', 'location_Shah Alam', 'location_Sibu', 'location_Sik', 'location_Solaris Dutamas', 'location_Sri Damansara', 'location_Sri Hartamas', 'location_Sri Kembangan', 'location_Sri Petaling', 'location_Subang', 'location_Subang Bestari', 'location_Subang Jaya', 'location_Sungai Besi', 'location_Sungai Buloh', 'location_Sungai Long', 'location_Sungai Pelek', 'location_Sungai Penchala', 'location_Sungai Way', 'location_Sunway', 'location_Sunway Spk', 'location_Taiping', 'location_Taman Bandar Utama', 'location_Taman Bukit Indah', 'location_Taman Bukit Puchong', 'location_Taman Cheras', 'location_Taman Cheras Indah', 'location_Taman Cheras Mutiara', 'location_Taman Cheras Perdana', 'location_Taman Connaught', 'location_Taman Desa', 'location_Taman Desa Aman', 'location_Taman Desa Jaya', 'location_Taman Desa Skudai', 'location_Taman Equine', 'location_Taman Kajang Perdana', 'location_Taman Maluri', 'location_Taman Melati', 'location_Taman Melawati', 'location_Taman Meranti Jaya', 'location_Taman OUG', 'location_Taman Perindustrian Puchong', 'location_Taman Permata', 'location_Taman Pinggiran Putra', 'location_Taman Puchong Indah', 'location_Taman Puchong Intan', 'location_Taman Puchong Perdana', 'location_Taman Puchong Perdana 2', 'location_Taman Puchong Permai 2', 'location_Taman Puchong Permai 3', 'location_Taman Puchong Prima', 'location_Taman Puchong Utama 4', 'location_Taman Puchong Utama Barat', 'location_Taman Putra Damai', 'location_Taman Putra Jaya', 'location_Taman Salak Selatan', 'location_Taman Sentosa', 'location_Taman Sentosa Perdana', 'location_Taman Sentul Jaya', 'location_Taman Seri Impian', 'location_Taman Seri Puchong', 'location_Taman Setapak Jaya', 'location_Taman Setia Indah', 'location_Taman Setia Tropika', 'location_Taman Setiawangsa', 'location_Taman Sri Manja', 'location_Taman Sri Muda', 'location_Taman Subang Mewah', 'location_Taman Sutera', 'location_Taman Tasik Puchong', 'location_Taman Tun Dr Ismail', 'location_Tanah Merah', 'location_Tanjong Duabelas', 'location_Tawau', 'location_Telok Panglima Garang', 'location_Temerloh', 'location_Titiwangsa', 'location_Tropicana', 'location_USJ', 'location_USJ 1', 'location_Ulu Kelang', 'location_Ulu Klang', 'location_Wangsa Maju', 'location_Wangsa Melawati', 'property_type_Apartment', 'property_type_Bungalow', 'property_type_Condominium', 'property_type_Flat', 'property_type_Semi-Detached House', 'property_type_Service Residence', 'property_type_Terraced House', 'property_type_Townhouse', 'tenure_Freehold', 'tenure_Leasehold', 'property_tier_Budget', 'property_tier_Mid-range', 'property_tier_Premium']

# --- Extract options dynamically ---
location_options = sorted([f.split("location_")[1] for f in all_features if f.startswith("location_")])
property_type_options = sorted([f.split("property_type_")[1] for f in all_features if f.startswith("property_type_")])
facility_options = [
    f for f in all_features
    if not f.startswith(("location_", "property_type_", "tenure_", "property_tier_"))
    and f not in ["size", "bedrooms", "bathrooms"]
]

# --- INPUTS ---
st.header("Property Details")

col1, col2, col3 = st.columns(3)
with col1:
    size = st.slider("Size (sqft)", min_value=500, max_value=5000, value=1500, step=50)
with col2:
    bedrooms = st.slider("Bedrooms", min_value=1, max_value=10, value=3)
with col3:
    bathrooms = st.slider("Bathrooms", min_value=1, max_value=6, value=2)

col1, col2, col3 = st.columns(3)
with col1:
    property_type = st.selectbox("Property Type", options=property_type_options)
with col2:
    location = st.selectbox("Location", options=location_options)
with col3:
    tenure = st.selectbox("Tenure Type", options=["Freehold", "Leasehold"])

st.subheader("Facilities")
cols = st.columns(4)
facilities = {}
for i, facility in enumerate(facility_options):
    with cols[i % 4]:
        facilities[facility] = st.checkbox(facility.replace("_", " ").title(), value=False)

# --- PREDICTIONS ---
if model is not None:
    # Create input data frame
    input_data = pd.DataFrame(0, index=[0], columns=all_features)
    input_data["size"] = forward_size_transform(size)
    input_data[["bedrooms", "bathrooms"]] = feature_scaler.transform(
        [[bedrooms, bathrooms]]
    )
    input_data[f"location_{location}"] = 1
    input_data[f"property_type_{property_type}"] = 1
    input_data[f"tenure_{tenure}"] = 1
    for facility, value in facilities.items():
        input_data[facility] = 1 if value else 0

    # Prediction (scaled log1p)
    dmatrix = xgb.DMatrix(input_data)
    scaled_log1p_prediction = model.predict(dmatrix)[0]

    # Inverse MinMaxScaler -> back to log1p(price)
    log1p_prediction = price_scaler.inverse_transform(
        np.array(scaled_log1p_prediction).reshape(-1, 1)
    )[0][0]

    # Reverse log1p -> back to RM
    final_prediction = reverse_log1p_transform(log1p_prediction)

    # Display result
    formatted_price = f"RM {final_prediction:,.2f}"
    st.markdown(
        f"""
        <div style="
            background-color: #D0E8FF;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            margin-top: 20px;
            margin-bottom: 20px;
        ">
            <h3 style="margin: 0; color: #555;">Predicted Property Price</h3>
            <h1 style="margin: 10px 0; color: #007acc;">{formatted_price}</h1>
            <p style="margin: 0; color: #888;">Based on your selected inputs</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- SHAP Explainability ---
    st.markdown("---")
    st.header("🔍 Explainability")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    # --- Filter to only user-selected features ---
    user_features = input_data.loc[:, (input_data != 0).any(axis=0)].columns
    user_indices = [input_data.columns.get_loc(f) for f in user_features]
    shap_values_user = shap_values[0][user_indices]

    # --- Plot bar chart (original) ---
    feature_importance = pd.DataFrame({
        "feature": user_features,
        "shap_value": shap_values_user,
        "abs_shap": np.abs(shap_values_user)
    }).sort_values("abs_shap", ascending=False)

    fig, ax = plt.subplots(figsize=(10,6))
    ax.barh(feature_importance["feature"], feature_importance["shap_value"])
    ax.set_xlabel("SHAP Value (Impact on Model Output)")
    ax.set_ylabel("Feature")
    ax.invert_yaxis()
    st.pyplot(fig)
    plt.close(fig)

    # --- Convert SHAP values to RM for textual explanation ---
    # Get predicted scaled log1p price
    dmatrix = xgb.DMatrix(input_data)
    scaled_log1p_pred = model.predict(dmatrix)[0]

    # Inverse MinMaxScaler to get log1p(price)
    log1p_pred = price_scaler.inverse_transform([[scaled_log1p_pred]])[0][0]
    final_pred = np.expm1(log1p_pred)  # predicted price in RM

    # Compute contribution of each feature in RM
    feature_contributions = pd.DataFrame({
        "feature": user_features,
        "shap_value_rm": shap_values_user / scaled_log1p_pred * final_pred
    }).sort_values("shap_value_rm", key=abs, ascending=False)

    # --- Display human-readable explanation ---
    st.subheader("📊 Feature Impact Analysis")

    cols = st.columns(3)
    for i, (_, row) in enumerate(feature_contributions.iterrows()):
        with cols[i % 3]:
            impact = row["shap_value_rm"]
            color = "green" if impact > 0 else "red"
            icon = "⬆️" if impact > 0 else "⬇️"
            
            st.markdown(f"""
            <div style="background-color: #F0F8FF; border: 1px solid {color}; border-radius: 10px; padding: 15px; margin: 10px 0;">
                <h4 style="color: {color}; margin: 0;">{icon} {row['feature'].replace('_', ' ').title()}</h4>
                <h3 style="color: {color}; margin: 5px 0;">RM {abs(impact):,.0f}</h3>
                <p style="margin: 0; color: #666;">{'Increased' if impact > 0 else 'Decreased'} value</p>
            </div>
            """, unsafe_allow_html=True)

    # --- Show detailed input data ---
    with st.expander("🔧 Show Detailed Input Data"):
        st.dataframe(input_data.loc[:, (input_data != 0).any(axis=0)])

else:
    st.error("❌ Model could not be loaded. Please check if 'xgboost_model.json' exists.")

# --- Sidebar info ---
with st.sidebar:
    st.header("About This App")
    st.markdown("""
    This predictive model estimates property prices based on:
    - **Physical attributes**: Size, bedrooms, bathrooms  
    - **Location**: Different areas have different values  
    - **Property type**: Various property types  
    - **Tenure**: Freehold vs. leasehold  
    - **Facilities**: Various amenities and facilities  

    The SHAP values help explain which factors most influenced the price prediction.
    """)
    st.header("How to Interpret Results")
    st.markdown("""
    - **Property Price**: Predicted market value in RM  
    - **SHAP values**: Show how much each feature increased/decreased the price  
    - **Positive values**: Increased the property price  
    - **Negative values**: Decreased the property price  
    - Values represent the approximate monetary impact (RM)
    """)
