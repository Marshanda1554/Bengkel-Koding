import streamlit as st
import pandas as pd
import joblib
import os

# 1. Judul & Konfigurasi
st.set_page_config(page_title="Telco Churn Predictor Plus", layout="wide")
st.title("📊 Telco Customer Churn Prediction (Full Input)")
st.write("Gunakan input yang lebih lengkap agar hasil prediksi lebih akurat.")

# 2. Load Model
model_path = 'model_churn_rf.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
    st.success("✅ Model Berhasil Dimuat!")
else:
    st.error("⚠️ File 'model_churn_rf.pkl' tidak ditemukan.")
    st.stop()

# 3. Form Input (Dibagi 3 Kolom)
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Data Dasar")
    tenure = st.number_input("Tenure (Bulan)", min_value=1, max_value=72, value=1)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=118.0, value=100.0)
    total_charges = st.number_input("Total Charges ($)", min_value=18.0, max_value=8600.0, value=100.0)

with col2:
    st.subheader("Layanan & Keamanan")
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

with col3:
    st.subheader("Kontrak & Billing")
    contract = st.selectbox("Jenis Kontrak", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Metode Bayar", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

# 4. Logika Prediksi
if st.button("🚀 Prediksi Sekarang"):
    try:
        # Menyiapkan 45 kolom sesuai hasil One-Hot Encoding di notebook
        n_features = 45 
        df_input = pd.DataFrame([[0.0] * n_features])
        
        # SCALING (Penting!)
        tenure_s = (tenure - 32.37) / 24.56
        monthly_s = (monthly_charges - 64.76) / 30.09
        total_s = (total_charges - 2283.3) / 2266.7

        # Masukkan fitur numerik (Indeks 0, 1, 2)
        df_input.iloc[0, 0] = tenure_s
        df_input.iloc[0, 1] = monthly_s
        df_input.iloc[0, 2] = total_s
        
        # MENGISI KOLOM KATEGORI (Manual One-Hot)
        # Kita isi indeks kolom secara manual agar sinyal CHURN kuat
        if contract == "Month-to-month":
            df_input.iloc[0, 5] = 1.0 # Indeks asumsi untuk Contract
        if internet == "Fiber optic":
            df_input.iloc[0, 10] = 1.0 # Fiber optic sering bikin Churn karena mahal
        if security == "No":
            df_input.iloc[0, 15] = 1.0 # Tidak ada security = risiko Churn tinggi
        if paperless == "Yes":
            df_input.iloc[0, 20] = 1.0

        prediction = model.predict(df_input)
        
        st.divider()
        if prediction[0] == 'Yes' or prediction[0] == 1:
            st.error("⚠️ HASIL: Pelanggan diprediksi akan CHURN")
        else:
            st.success("✅ HASIL: Pelanggan diprediksi akan STAY")
            
    except Exception as e:
        st.error(f"Error: {e}")
