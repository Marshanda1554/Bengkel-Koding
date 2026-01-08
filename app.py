import streamlit as st
import pandas as pd
import pickle
import os

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Telco Churn Predictor", layout="wide")
st.title("📊 Telco Customer Churn Prediction")

# 2. Load Model menggunakan Pickle Standar
model_path = 'model_churn_rf.pkl'
if os.path.exists(model_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        st.success("✅ Model Berhasil Dimuat!")
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        st.stop()
else:
    st.error(f"⚠️ File '{model_path}' tidak ditemukan.")
    st.stop()

# 3. Form Input (Urutan harus persis sesuai dataset training)
st.divider()
st.header("📝 Masukkan Data Pelanggan")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox("SeniorCitizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.number_input("tenure", min_value=0, max_value=100, value=1)

with col2:
    PhoneService = st.selectbox("PhoneService", ["No", "Yes"])
    MultipleLines = st.selectbox("MultipleLines", ["No phone service", "No", "Yes"])
    InternetService = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("OnlineSecurity", ["No", "Yes", "No internet service"])
    OnlineBackup = st.selectbox("OnlineBackup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("DeviceProtection", ["No", "Yes", "No internet service"])

with col3:
    TechSupport = st.selectbox("TechSupport", ["No", "Yes", "No internet service"])
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
        # Menyesuaikan input ke DataFrame
        input_dict = {
            'gender': gender, 'SeniorCitizen': SeniorCitizen, 'Partner': Partner,
            'Dependents': Dependents, 'tenure': tenure, 'PhoneService': PhoneService,
            'MultipleLines': MultipleLines, 'InternetService': InternetService,
            'OnlineSecurity': OnlineSecurity, 'OnlineBackup': OnlineBackup,
            'DeviceProtection': DeviceProtection, 'TechSupport': TechSupport,
            'StreamingTV': StreamingTV, 'StreamingMovies': StreamingMovies,
            'Contract': Contract, 'PaperlessBilling': PaperlessBilling,
            'PaymentMethod': PaymentMethod, 'MonthlyCharges': MonthlyCharges,
            'TotalCharges': TotalCharges
        }
        input_df = pd.DataFrame([input_dict])

        prediction = model.predict(input_df)
        
        st.divider()
        if prediction[0] == 'Yes' or prediction[0] == 1:
            st.error("⚠️ HASIL: Pelanggan diprediksi akan CHURN")
        else:
            st.success("✅ HASIL: Pelanggan diprediksi akan STAY")
            
    except Exception as e:
        st.error(f"Kesalahan Prediksi: {e}")
        
