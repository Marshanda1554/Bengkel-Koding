import streamlit as st
import pandas as pd
import joblib
import os

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Telco Churn Predictor", layout="wide")
st.title("📊 Aplikasi Prediksi Churn Pelanggan")
st.write("Aplikasi ini menggunakan model Pipeline yang sudah divalidasi dengan Cross-Validation.")

# 2. Load Model (Pastikan model yang disimpan adalah Pipeline 'best_rf_model')
model_path = 'model_churn_rf.pkl'
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        st.success("✅ Model Pipeline Berhasil Dimuat!")
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()
else:
    st.error("⚠️ File 'model_churn_rf.pkl' tidak ditemukan.")
    st.stop()

# 3. Form Input Data (Dibuat lengkap agar model tidak bingung)
st.divider()
st.subheader("📝 Masukkan Data Pelanggan")

col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.number_input("Tenure (Bulan)", min_value=1, max_value=72, value=1)
    monthly = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=100.0)
    total = st.number_input("Total Charges ($)", min_value=18.0, max_value=9000.0, value=100.0)
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])

with col2:
    contract = st.selectbox("Jenis Kontrak", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

with col3:
    senior = st.selectbox("Senior Citizen (Lansia)", [0, 1])
    partner = st.selectbox("Memiliki Pasangan", ["No", "Yes"])
    dependents = st.selectbox("Memiliki Tanggungan", ["No", "Yes"])
    billing = st.selectbox("Paperless Billing", ["Yes", "No"])

# 4. Logika Prediksi
if st.button("🚀 Prediksi Sekarang"):
    try:
        # Kita buat DataFrame dengan 19 kolom ASLI (Bukan 45 kolom manual)
        # Karena Pipeline kamu akan melakukan scaling dan encoding sendiri secara otomatis!
        input_data = pd.DataFrame({
            'gender': [gender],
            'SeniorCitizen': [senior],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': ["Yes"], # Default
            'MultipleLines': ["No"], # Default
            'InternetService': [internet],
            'OnlineSecurity': [security],
            'OnlineBackup': ["No"], # Default
            'DeviceProtection': ["No"], # Default
            'TechSupport': [tech_support],
            'StreamingTV': ["No"], # Default
            'StreamingMovies': ["No"], # Default
            'Contract': [contract],
            'PaperlessBilling': [billing],
            'PaymentMethod': ["Electronic check"], # Default
            'MonthlyCharges': [monthly],
            'TotalCharges': [total]
        })

        # Prediksi langsung menggunakan pipeline (model akan preprocess sendiri)
        prediction = model.predict(input_data)
        
        st.divider()
        st.subheader("🔍 Hasil Prediksi:")
        
        # Sesuai target di notebook kamu (Churn: Yes/No)
        if prediction[0] == 'Yes' or prediction[0] == 1:
            st.error("⚠️ HASIL: Pelanggan diprediksi akan CHURN (Berhenti)")
            st.write("Alasan Teknis: Kontrak bulanan dan layanan Fiber Optic tanpa keamanan tambahan meningkatkan risiko pindah provider.")
        else:
            st.success("✅ HASIL: Pelanggan diprediksi akan STAY (Bertahan)")
            st.write("Alasan Teknis: Profil ini menunjukkan loyalitas yang stabil.")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
        st.info("Pastikan model yang diupload adalah 'best_rf_model' atau pipeline yang sudah terlatih.")

st.divider()
st.caption("Aplikasi Prediksi Churn - Bengkel Koding UAS")
