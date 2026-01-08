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

# 2. Input Sederhana
st.divider()
tenure = st.number_input("Tenure (Bulan)", min_value=0, value=1)
monthly = st.number_input("Monthly Charges ($)", value=70.0)
total = st.number_input("Total Charges ($)", value=70.0)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

if st.button("🚀 Prediksi Sekarang"):
    try:
        # KUNCI JAWABAN: Kita susun DataFrame dengan urutan yang berbeda
        # Kita taruh angka di depan karena Scaler biasanya nunggu di kolom 0, 1, 2
        input_data = pd.DataFrame({
            'tenure': [tenure],
            'MonthlyCharges': [monthly],
            'TotalCharges': [total],
            'gender': ['Male'],
            'SeniorCitizen': [0],
            'Partner': ['No'],
            'Dependents': ['No'],
            'PhoneService': ['Yes'],
            'MultipleLines': ['No'],
            'InternetService': ['Fiber optic'],
            'OnlineSecurity': ['No'],
            'OnlineBackup': ['No'],
            'DeviceProtection': ['No'],
            'TechSupport': ['No'],
            'StreamingTV': ['No'],
            'StreamingMovies': ['No'],
            'Contract': [contract],
            'PaperlessBilling': ['Yes'],
            'PaymentMethod': ['Electronic check']
        })

        # Coba prediksi
        prediction = model.predict(input_data)
        
        st.divider()
        if prediction[0] == 'Yes' or prediction[0] == 1:
            st.error("⚠️ HASIL: CHURN")
        else:
            st.success("✅ HASIL: STAY")
            
    except Exception as e:
        # Jika masih error, kita coba balik urutannya otomatis di sini
        try:
            # Versi cadangan: Geser angka ke paling belakang
            cols = list(input_data.columns)
            input_data_v2 = input_data[cols[3:] + cols[:3]]
            prediction = model.predict(input_data_v2)
            st.write("Hasil (v2): " + str(prediction[0]))
        except:
            st.error(f"Model kamu minta urutan khusus. Error: {e}")
