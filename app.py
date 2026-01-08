import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Telco Churn Predictor", layout="centered")
st.title("📊 Telco Customer Churn Prediction")
st.write("A11.2022.14816 - Marshanda Putri Salsabila")

# 1. Load Model
model_path = 'model_churn_rf.pkl'
@st.cache_resource
def load_model():
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

if model is None:
    st.error("⚠️ Model tidak ditemukan!")
    st.stop()

# 2. Input yang paling minimalis
st.divider()
col1, col2 = st.columns(2)
with col1:
    tenure = st.number_input("Tenure (Bulan)", min_value=0, value=1)
    monthly = st.number_input("Monthly Charges ($)", value=70.0)
    total = st.number_input("Total Charges ($)", value=70.0)
with col2:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])

if st.button("🚀 Prediksi Sekarang"):
    try:
        # KITA BUAT SEMUA KOLOM (19 KOLOM) 
        # TAPI KITA MASUKKAN KE LIST AGAR URUTANNYA KAKU
        # Urutan: numerik dulu baru kategorikal (Ini standar Pipeline Scikit-Learn)
        
        data_numerik = [tenure, monthly, total, 0] # 0 itu untuk SeniorCitizen
        data_kategorikal = [
            'Male', 'No', 'No', 'Yes', 'No', internet, 'No', 'No', 
            'No', 'No', 'No', 'No', contract, 'Yes', 'Electronic check'
        ]
        
        # Gabungkan semua (Total 19 fitur)
        full_data = data_numerik + data_kategorikal
        
        # Buat DataFrame dengan nama kolom standar dataset Telco
        kolom = [
            'tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen',
            'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod'
        ]
        
        input_df = pd.DataFrame([full_data], columns=kolom)

        # Prediksi
        prediction = model.predict(input_df)
        
        st.divider()
        if prediction[0] == 'Yes' or prediction[0] == 1:
            st.error("⚠️ HASIL: CHURN")
        else:
            st.success("✅ HASIL: STAY")
            
    except Exception as e:
        st.error(f"Urutan Model masih bentrok. Pesan: {e}")
        st.info("Tips: Saat presentasi, fokus pada EDA di Notebook jika web ini masih terkendala metadata.")
