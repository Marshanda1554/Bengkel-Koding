import streamlit as st
import pandas as pd
import joblib
import os

# 1. Judul
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
    st.success("✅ Model Siap Digunakan!")
else:
    st.error("⚠️ Model tidak ditemukan. Pastikan file .pkl sudah di GitHub.")
    st.stop()

# 3. Form Input (Tanpa Gender & Data Ribet Lainnya)
st.divider()
st.header("📝 Masukkan Data Pelanggan")
col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (Sudah berapa bulan berlangganan?)", min_value=0, value=1)
    contract = st.selectbox("Jenis Kontrak", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Layanan Internet", ["Fiber optic", "DSL", "No"])
    monthly_charges = st.number_input("Tagihan Bulanan ($)", value=70.0)

with col2:
    security = st.selectbox("Keamanan Online (Online Security)", ["No", "Yes", "No internet service"])
    support = st.selectbox("Bantuan Teknis (Tech Support)", ["No", "Yes", "No internet service"])
    billing = st.selectbox("Tagihan Tanpa Kertas (Paperless Billing)", ["Yes", "No"])
    total_charges = st.number_input("Total Tagihan ($)", value=70.0)

# 4. Prediksi
if st.button("🚀 Prediksi Sekarang"):
    try:
        # TRIK: Kita isi data sisanya otomatis agar model tidak error
        input_data = pd.DataFrame([{
            'gender': 'Male',           # Diisi otomatis (Hardcoded)
            'SeniorCitizen': 0,         # Diisi otomatis
            'Partner': 'No',            # Diisi otomatis
            'Dependents': 'No',         # Diisi otomatis
            'tenure': tenure,
            'PhoneService': 'Yes',      # Diisi otomatis
            'MultipleLines': 'No',      # Diisi otomatis
            'InternetService': internet,
            'OnlineSecurity': security,
            'OnlineBackup': 'No',       # Diisi otomatis
            'DeviceProtection': 'No',   # Diisi otomatis
            'TechSupport': support,
            'StreamingTV': 'No',        # Diisi otomatis
            'StreamingMovies': 'No',    # Diisi otomatis
            'Contract': contract,
            'PaperlessBilling': billing,
            'PaymentMethod': 'Electronic check', # Diisi otomatis
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }])

        # Pastikan urutan kolom sesuai standar dataset Telco 
        prediction = model.predict(input_data)
        
        st.divider()
        if prediction[0] == 'Yes' or prediction[0] == 1:
            st.error("⚠️ HASIL: Pelanggan diprediksi akan CHURN (Berhenti)")
        else:
            st.success("✅ HASIL: Pelanggan diprediksi akan STAY (Bertahan)")
            
    except Exception as e:
        st.error(f"Kesalahan: {e}")
