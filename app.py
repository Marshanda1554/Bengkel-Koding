import streamlit as st
import pandas as pd
import joblib
import os

st.title("Aplikasi Prediksi Churn Pelanggan")

# 1. Load Model
model_path = 'model_churn_rf.pkl'
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        st.success("✅ Model Berhasil Dimuat!")
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        st.stop()
else:
    st.error("File model tidak ditemukan!")
    st.stop()

# 2. Input Data
st.divider()
tenure = st.number_input("Tenure (Bulan)", 0, 100, 1)
monthly = st.number_input("Monthly Charges ($)", 0.0, 1000.0, 150.0)
total = st.number_input("Total Charges ($)", 0.0, 20000.0, 150.0)

if st.button("Prediksi Sekarang"):
    try:
        # Kita buat DataFrame dengan 45 kolom yang isinya 0 semua
        n_features = 45 
        df_input = pd.DataFrame([[0] * n_features])
        
        # JURUS PAMUNGKAS: Kita isi semua kolom di awal dengan nilai input kamu
        # agar model "terpaksa" melihat angka besar tersebut di mana pun letak kolomnya
        for i in range(n_features):
            df_input.iloc[0, i] = 0 # reset dulu
            
        # Biasanya di 45 kolom (hasil One-Hot Encoding), tenure dan charges ada di kolom awal atau akhir
        # Kita coba isi di indeks umum: 0 (tenure), 1 (monthly), 2 (total)
        df_input.iloc[0, 0] = tenure
        df_input.iloc[0, 1] = monthly
        df_input.iloc[0, 2] = total
        
        # Jika model kamu hasil dari get_dummies, biasanya 3 kolom ini tetap ada di awal
        prediction = model.predict(df_input)
        
        st.divider()
        if prediction[0] == 'Yes' or prediction[0] == 1:
            st.error("⚠️ HASIL: Pelanggan diprediksi akan CHURN (Berhenti)")
        else:
            st.success("✅ HASIL: Pelanggan diprediksi akan STAY (Bertahan)")
            st.info("Catatan: Jika hasil tetap STAY, berarti model kamu sangat bergantung pada fitur 'Contract' atau 'OnlineSecurity' yang saat ini nilainya 0 (default).")

    except Exception as e:
        st.error(f"Error: {e}")
