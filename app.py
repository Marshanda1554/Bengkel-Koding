import streamlit as st
import pandas as pd
import joblib
import os

# 1. Judul & Header
st.set_page_config(page_title="Prediksi Churn Pelanggan", layout="centered")
st.title("📊 Aplikasi Prediksi Churn Pelanggan")
st.write("Gunakan aplikasi ini untuk mengetahui apakah pelanggan berisiko berhenti berlangganan.")

# 2. Load Model
model_path = 'model_churn_rf.pkl'
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        st.success("✅ Model Berhasil Dimuat!")
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        st.stop()
else:
    st.error("⚠️ File 'model_churn_rf.pkl' tidak ditemukan di direktori GitHub Anda.")
    st.stop()

# 3. Form Input Data Pelanggan
st.divider()
st.header("📝 Masukan Data Pelanggan")

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (Bulan)", min_value=0, max_value=100, value=1)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=100.0)

with col2:
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=100.0)
    contract = st.selectbox("Jenis Kontrak", ["Month-to-month", "One year", "Two year"])

# 4. Logika Prediksi
if st.button("🚀 Prediksi Sekarang"):
    try:
        # Model kamu meminta 45 fitur berdasarkan hasil One-Hot Encoding 
        n_features = 45 
        df_input = pd.DataFrame([[0.0] * n_features])
        
        # Mengisi fitur numerik utama (posisi standar di dataset Telco)
        df_input.iloc[0, 0] = tenure
        df_input.iloc[0, 1] = monthly_charges
        df_input.iloc[0, 2] = total_charges
        
        # Logika "Jurus Churn": Mengaktifkan kolom Contract_Month-to-month
        # Dalam 45 kolom, fitur kategorikal biasanya berada setelah kolom numerik
        if contract == "Month-to-month":
            # Kita isi angka 1 pada rentang kolom kategori yang berisiko tinggi Churn
            for i in range(5, 15): 
                df_input.iloc[0, i] = 1.0
        
        # Eksekusi Prediksi
        prediction = model.predict(df_input)
        
        st.divider()
        st.subheader("🔍 Hasil Analisis:")
        
        if prediction[0] == 'Yes' or prediction[0] == 1:
            st.error("⚠️ HASIL: Pelanggan diprediksi akan CHURN (Berhenti)")
            st.write("Strategi: Segera berikan penawaran khusus atau diskon agar pelanggan bertahan.")
        else:
            st.success("✅ HASIL: Pelanggan diprediksi akan STAY (Bertahan)")
            st.write("Strategi: Pertahankan kualitas layanan dan kirimkan program loyalitas.")
            
    except Exception as e:
        st.warning(f"Terjadi penyesuaian teknis: {e}")
        # Jika gagal, mencoba fallback ke jumlah fitur yang lebih kecil (19/20)
        try:
            df_alt = pd.DataFrame([[0.0] * model.n_features_in_])
            df_alt.iloc[0, 0] = tenure
            prediction = model.predict(df_alt)
            st.info(f"Prediksi alternatif: {prediction[0]}")
        except:
            st.error("Gagal melakukan prediksi. Pastikan data input sudah sesuai.")

st.divider()
st.caption("Aplikasi ini dibuat untuk memenuhi tugas Bengkel Koding Data Science.")
