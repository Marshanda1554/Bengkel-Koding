import streamlit as st
import pandas as pd
import joblib
import os

# 1. Judul & Konfigurasi
st.set_page_config(page_title="Telco Churn Predictor", layout="wide")
st.title("📊 Telco Customer Churn Prediction")

# 2. Load Model
# Karena di notebook kamu pakai 'dill' untuk simpan, di sini kita coba load dengan joblib
# Jika masih error, pastikan file .pkl yang kamu upload adalah hasil dari cell terakhir di notebook.
model_path = 'model_churn_rf.pkl'
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        st.success("✅ Model Pipeline Berhasil Dimuat!")
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()
else:
    st.error(f"⚠️ File '{model_path}' tidak ditemukan.")
    st.stop()

# 3. Form Input Data (URUTAN HARUS SAMA DENGAN X_train.columns)
st.divider()
st.header("📝 Masukkan Data Pelanggan")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox("Senior Citizen (Lansia)", [0, 1])
    Partner = st.selectbox("Partner (Punya Pasangan)", ["Yes", "No"])
    Dependents = st.selectbox("Dependents (Punya Tanggungan)", ["No", "Yes"])
    tenure = st.number_input("Tenure (Bulan)", min_value=1, max_value=72, value=1)
    PhoneService = st.selectbox("Phone Service", ["No", "Yes"])

with col2:
    MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

with col3:
    StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    MonthlyCharges = st.number_input("Monthly Charges ($)", value=100.0)
    TotalCharges = st.number_input("Total Charges ($)", value=100.0)

# 4. Logika Prediksi
if st.button("🚀 Prediksi Sekarang"):
    try:
        # MEMBUAT DATAFRAME DENGAN URUTAN DAN TIPE DATA YANG PAS
        input_data = pd.DataFrame({
            'gender': [gender],
            'SeniorCitizen': [int(SeniorCitizen)],
            'Partner': [Partner],
            'Dependents': [Dependents],
            'tenure': [int(tenure)],
            'PhoneService': [PhoneService],
            'MultipleLines': [MultipleLines],
            'InternetService': [InternetService],
            'OnlineSecurity': [OnlineSecurity],
            'OnlineBackup': [OnlineBackup],
            'DeviceProtection': [DeviceProtection],
            'TechSupport': [TechSupport],
            'StreamingTV': [StreamingTV],
            'StreamingMovies': [StreamingMovies],
            'Contract': [Contract],
            'PaperlessBilling': [PaperlessBilling],
            'PaymentMethod': [PaymentMethod],
            'MonthlyCharges': [float(MonthlyCharges)],
            'TotalCharges': [float(TotalCharges)]
        })

        # Prediksi (Pipeline akan otomatis handle Scaling dan One-Hot Encoding)
        prediction = model.predict(input_data)
        
        st.divider()
        if prediction[0] == 'Yes' or prediction[0] == 1:
            st.error("⚠️ HASIL: Pelanggan diprediksi akan CHURN")
        else:
            st.success("✅ HASIL: Pelanggan diprediksi akan STAY")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
        st.info("Pesan ini muncul karena model mengharapkan input dalam urutan tertentu.")

st.divider()
st.caption("A11.2022.14816 - Marshanda Putri Salsabila")
