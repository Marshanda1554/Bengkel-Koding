import streamlit as st
import pandas as pd
import joblib
import os

# 1. Judul & Konfigurasi
st.set_page_config(page_title="Telco Churn Predictor", layout="wide")
st.title("📊 Telco Customer Churn Prediction")
st.write("A11.2022.14816 - Marshanda Putri Salsabila")

# 2. Load Model
model_path = 'model_churn_rf.pkl'

@st.cache_resource
def load_model():
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

if model is not None:
    st.success("✅ Model Berhasil Dimuat!")
else:
    st.error("⚠️ Model tidak ditemukan. Pastikan 'model_churn_rf.pkl' ada di GitHub.")
    st.stop()

# 3. Form Input (Urutan ini harus 100% sama dengan urutan kolom di X_train)
st.divider()
st.header("📝 Masukkan Data Pelanggan")
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox("SeniorCitizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.number_input("tenure", min_value=0, value=1)
    PhoneService = st.selectbox("PhoneService", ["No", "Yes"])

with col2:
    MultipleLines = st.selectbox("MultipleLines", ["No phone service", "No", "Yes"])
    InternetService = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("OnlineSecurity", ["No", "Yes", "No internet service"])
    OnlineBackup = st.selectbox("OnlineBackup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("DeviceProtection", ["No", "Yes", "No internet service"])
    TechSupport = st.selectbox("TechSupport", ["No", "Yes", "No internet service"])

with col3:
    StreamingTV = st.selectbox("StreamingTV", ["No", "Yes", "No internet service"])
    StreamingMovies = st.selectbox("StreamingMovies", ["No", "Yes", "No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("PaperlessBilling", ["Yes", "No"])
    PaymentMethod = st.selectbox("PaymentMethod", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    MonthlyCharges = st.number_input("MonthlyCharges", value=70.0)
    TotalCharges = st.number_input("TotalCharges", value=70.0)

# 4. Prediksi
if st.button("🚀 Prediksi Sekarang"):
    try:
        # DATA DIBAWAH INI HARUS SESUAI URUTAN FITUR ASLI
        # Berdasarkan dataset Telco: gender, SeniorCitizen, Partner, Dependents, tenure...
        input_data = pd.DataFrame([{
            'gender': gender,
            'SeniorCitizen': SeniorCitizen,
            'Partner': Partner,
            'Dependents': Dependents,
            'tenure': tenure,
            'PhoneService': PhoneService,
            'MultipleLines': MultipleLines,
            'InternetService': InternetService,
            'OnlineSecurity': OnlineSecurity,
            'OnlineBackup': OnlineBackup,
            'DeviceProtection': DeviceProtection,
            'TechSupport': TechSupport,
            'StreamingTV': StreamingTV,
            'StreamingMovies': StreamingMovies,
            'Contract': Contract,
            'PaperlessBilling': PaperlessBilling,
            'PaymentMethod': PaymentMethod,
            'MonthlyCharges': MonthlyCharges,
            'TotalCharges': TotalCharges
        }])

        # Pipeline akan mengarahkan kolom numerik ke Scaler dan kategorikal ke OneHot secara otomatis
        prediction = model.predict(input_data)
        
        st.divider()
        if prediction[0] == 'Yes' or prediction[0] == 1:
            st.error("⚠️ HASIL: Pelanggan diprediksi akan CHURN (Berhenti)")
        else:
            st.success("✅ HASIL: Pelanggan diprediksi akan STAY (Bertahan)")
            
    except Exception as e:
        st.error(f"Kesalahan: {e}")
