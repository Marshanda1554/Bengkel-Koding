import streamlit as st
import pandas as pd
import joblib
import os

# 1. Judul & Konfigurasi
st.set_page_config(page_title="Telco Churn Predictor", layout="wide")
st.title("📊 Telco Customer Churn Prediction")

# 2. Load Model
model_path = 'model_churn_rf.pkl'
if os.path.exists(model_path):
    try:
        # Gunakan model pipeline yang sudah kamu simpan di notebook
        model = joblib.load(model_path)
        st.success("✅ Model Pipeline Berhasil Dimuat!")
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()
else:
    st.error(f"⚠️ File '{model_path}' tidak ditemukan.")
    st.stop()

# 3. Form Input Data (Urutan disesuaikan dengan dataset asli)
st.divider()
st.header("📝 Masukkan Data Pelanggan")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox("SeniorCitizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.number_input("tenure", min_value=1, max_value=72, value=1)

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
    MonthlyCharges = st.number_input("MonthlyCharges", value=100.0)
    TotalCharges = st.number_input("TotalCharges", value=100.0)

# 4. Logika Prediksi
if st.button("🚀 Prediksi Sekarang"):
    try:
        # MEMBUAT DATAFRAME DENGAN URUTAN KOLOM YANG BENAR
        # Urutan ini harus sama dengan df_clean.drop(['Churn', 'customerID'], axis=1)
        input_dict = {
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
        }
        
        input_df = pd.DataFrame([input_dict])

        # Prediksi menggunakan Pipeline (Pipeline otomatis handle Scaling & OneHot)
        prediction = model.predict(input_df)
        
        st.divider()
        if prediction[0] == 'Yes' or prediction[0] == 1:
            st.error("⚠️ HASIL: Pelanggan diprediksi akan CHURN")
        else:
            st.success("✅ HASIL: Pelanggan diprediksi akan STAY")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
        st.info("Saran: Pastikan urutan fitur di input_df sama dengan saat training di notebook.")

st.divider()
st.caption("A11.2022.14816 - Marshanda Putri Salsabila")
