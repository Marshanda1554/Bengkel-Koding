import streamlit as st
import pandas as pd
import dill
import os

# 1. Judul & Konfigurasi
st.set_page_config(page_title="Telco Churn Predictor", layout="wide")
st.title("📊 Telco Customer Churn Prediction")
st.write("A11.2022.14816 - Marshanda Putri Salsabila")

# 2. Load Model Menggunakan DILL
model_path = 'model_churn_rf.pkl'

@st.cache_resource # Agar model tidak di-load berulang-ulang setiap klik
def load_model():
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return dill.load(f)
    return None

try:
    model = load_model()
    if model is not None:
        st.success("✅ Model Pipeline Berhasil Dimuat!")
    else:
        st.error(f"⚠️ File '{model_path}' tidak ditemukan di GitHub.")
        st.stop()
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.warning("Tips: Jika error 'STACK_GLOBAL' muncul, hapus aplikasi di Streamlit Cloud lalu 'New App' kembali agar library terupdate.")
    st.stop()

# 3. Form Input Data (Urutan Sesuai Dataset Asli)
st.divider()
st.header("📝 Masukkan Data Pelanggan")
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.number_input("Tenure (Bulan)", min_value=0, value=1)

with col2:
    PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
    MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

with col3:
    TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    MonthlyCharges = st.number_input("Monthly Charges ($)", value=70.0)
    TotalCharges = st.number_input("Total Charges ($)", value=70.0)

# 4. Prediksi
if st.button("🚀 Prediksi Sekarang"):
    try:
        input_df = pd.DataFrame([{
            'gender': gender, 'SeniorCitizen': SeniorCitizen, 'Partner': Partner,
            'Dependents': Dependents, 'tenure': tenure, 'PhoneService': PhoneService,
            'MultipleLines': MultipleLines, 'InternetService': InternetService,
            'OnlineSecurity': OnlineSecurity, 'OnlineBackup': OnlineBackup,
            'DeviceProtection': DeviceProtection, 'TechSupport': TechSupport,
            'StreamingTV': StreamingTV, 'StreamingMovies': StreamingMovies,
            'Contract': Contract, 'PaperlessBilling': PaperlessBilling,
            'PaymentMethod': PaymentMethod, 'MonthlyCharges': MonthlyCharges,
            'TotalCharges': TotalCharges
        }])

        prediction = model.predict(input_df)
        
        st.divider()
        if prediction[0] == 'Yes' or prediction[0] == 1:
            st.error("⚠️ HASIL: Pelanggan diprediksi akan CHURN (Berhenti)")
        else:
            st.success("✅ HASIL: Pelanggan diprediksi akan STAY (Bertahan)")
            
    except Exception as e:
        st.error(f"Kesalahan Prediksi: {e}")
