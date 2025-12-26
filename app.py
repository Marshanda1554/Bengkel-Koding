import streamlit as st
import pandas as pd
import joblib
import os

# 1. Judul Aplikasi
st.title("Aplikasi Prediksi Churn Pelanggan Telco")
st.write("Dibuat oleh: Marshanda Putri Salsabila")

# 2. Load Model yang sudah didownload
# Kode untuk melacak file
st.write("Lokasi folder saat ini:", os.getcwd())
st.write("Daftar file yang terbaca oleh Streamlit:", os.listdir('.'))

# Coba muat model
nama_file = 'model_churn_rf.pkl'

if os.path.exists(nama_file):
    model = joblib.load(model_churn_rf.pkl)
    st.success("Berhasil: Model ditemukan dan dimuat!")
else:
    st.error(f"Gagal: File {model_churn_rf.pkl} TIDAK ADA di folder GitHub kamu.")

st.divider()

# 3. Input Data Pelanggan (Sesuaikan dengan fitur saat training)
st.header("Input Data Pelanggan")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (Bulan)", 0, 72, 12)

with col2:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
    monthly = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
    total = st.number_input("Total Charges ($)", min_value=0.0, value=500.0)

# 4. Tombol Prediksi
if st.button("Cek Prediksi"):
    # Buat DataFrame dari input user
    data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [senior],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'Contract': [contract],
        'PaperlessBilling': [paperless],
        'PaymentMethod': [payment],
        'MonthlyCharges': [monthly],
        'TotalCharges': [total]
    })
    
    # Tambahkan kolom lain yang ada saat training dengan nilai default (misal Service dll)
    # Ini penting agar jumlah kolom input sama dengan saat training model
    # (Catatan: Tambahkan kolom lain sesuai struktur X_train kamu)

    try:
        prediction = model.predict(data)
        
        if prediction[0] == 'Yes':
            st.error("HASIL: Pelanggan diprediksi akan CHURN (Berhenti)")
        else:
            st.success("HASIL: Pelanggan diprediksi akan STAY (Tetap)")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")



        




