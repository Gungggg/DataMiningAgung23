import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Tsunami AI Detector",
    page_icon="🌊",
    layout="wide"
)

# ==========================================
# 1. LOAD MODEL & TOOLS
# ==========================================
@st.cache_resource
def load_models():
    try:
        model_rf = joblib.load('model_rf.pkl')
        model_gb = joblib.load('model_gb.pkl')
        model_ensemble = joblib.load('model_ensemble.pkl')
        tools = joblib.load('preprocessing_tools.pkl')
        return model_rf, model_gb, model_ensemble, tools
    except FileNotFoundError:
        st.error("File .pkl tidak ditemukan! Pastikan Anda sudah menjalankan kode training (langkah sebelumnya).")
        return None, None, None, None

rf_model, gb_model, ensemble_model, tools = load_models()

# Jika model gagal dimuat, hentikan aplikasi
if tools is None:
    st.stop()

scaler = tools['scaler']
feature_names = tools['feature_names']

# ==========================================
# 2. SIDEBAR (METODE & PERFORMA)
# ==========================================
st.sidebar.title("⚙️ Konfigurasi Model")
st.sidebar.title("@copyright Agung Setyadi - 2023")
# Pilihan Model (Metode B: Head-to-Head)
model_choice = st.sidebar.selectbox(
    "Pilih Algoritma Cerdas:",
    ["Ensemble Voting (Akurasi Tertinggi)", "Random Forest", "Gradient Boosting"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Performa Model (Test Data)")

# Menampilkan performa berdasarkan pilihan (Hardcoded dari hasil training sebelumnya agar user lihat)
if model_choice == "Ensemble Voting (Akurasi Tertinggi)":
    st.sidebar.metric(label="Akurasi", value="91.72%", delta="Superior")
    active_model = ensemble_model
    st.sidebar.info("Gabungan kekuatan Random Forest & Gradient Boosting.")
elif model_choice == "Random Forest":
    st.sidebar.metric(label="Akurasi", value="89.80%", delta="-1.9%")
    active_model = rf_model
elif model_choice == "Gradient Boosting":
    st.sidebar.metric(label="Akurasi", value="92.35%", delta="+0.6%")
    active_model = gb_model

st.sidebar.markdown("---")
st.sidebar.markdown("**Tentang Data:**")
st.sidebar.caption("Menggunakan parameter fisika gempa (Magnitude, Kedalaman, Intensitas, dll) untuk memprediksi potensi Tsunami.")

# ==========================================
# 3. HALAMAN UTAMA (INPUT DATA)
# ==========================================
st.title("🌊 Tsunami Early Warning AI")
st.markdown("""
Aplikasi ini menggunakan **Machine Learning (Ensemble Method)** untuk memprediksi potensi tsunami berdasarkan data gempa bumi secara real-time.
""")

st.markdown("### 📝 Masukkan Parameter Gempa")

# Membuat Form Input yang Rapi dengan Kolom
with st.form("input_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Lokasi & Kekuatan")
        magnitude = st.number_input("Magnitude (Skala Richter)", min_value=1.0, max_value=10.0, value=7.0, step=0.1)
        depth = st.number_input("Kedalaman (km)", min_value=0.0, value=20.0, help="Semakin dangkal, potensi tsunami biasanya semakin besar.")
        latitude = st.number_input("Latitude", value=-9.79)
        longitude = st.number_input("Longitude", value=159.59)

    with col2:
        st.subheader("Intensitas")
        mmi = st.slider("MMI (Modified Mercalli Intensity)", 1, 12, 7, help="Tingkat guncangan yang dirasakan.")
        cdi = st.slider("CDI (Community Decimal Intensity)", 0, 12, 8)
        sig = st.number_input("Significance (Tingkat Signifikansi)", value=768, help="Nilai signifikansi kejadian gempa.")

    with col3:
        st.subheader("Teknis Seismik")
        nst = st.number_input("NST (Jumlah Stasiun Sensor)", value=117)
        dmin = st.number_input("Dmin (Jarak Stasiun Terdekat)", value=0.5)
        gap = st.number_input("Gap (Celah Azimuth)", value=17.0)

    # Tombol Submit
    submitted = st.form_submit_button("🔍 Analisis Potensi Tsunami")

# ==========================================
# 4. LOGIKA PREDIKSI
# ==========================================
if submitted:
    # 1. Bungkus data input ke DataFrame sesuai urutan training
    # Urutan kolom: ['magnitude', 'cdi', 'mmi', 'sig', 'nst', 'dmin', 'gap', 'depth', 'latitude', 'longitude']
    input_data = pd.DataFrame([[
        magnitude, cdi, mmi, sig, nst, dmin, gap, depth, latitude, longitude
    ]], columns=feature_names)

    # 2. Preprocessing (Scaling)
    # Kita gunakan scaler yang sama saat training
    try:
        input_scaled = scaler.transform(input_data)
    except Exception as e:
        st.error(f"Terjadi kesalahan preprocessing: {e}")
        st.stop()

    # 3. Prediksi
    prediction = active_model.predict(input_scaled)[0]
    probability = active_model.predict_proba(input_scaled)[0]

    # ==========================================
    # 5. TAMPILAN HASIL (RESULT)
    # ==========================================
    st.markdown("---")
    st.subheader("🎯 Hasil Analisis AI")

    col_res1, col_res2 = st.columns([1, 2])

    with col_res1:
        # Tampilkan Probabilitas dalam Chart
        st.write("Tingkat Keyakinan Model:")
        prob_df = pd.DataFrame({
            'Kondisi': ['Aman', 'Tsunami'],
            'Probabilitas': probability
        })
        st.bar_chart(prob_df.set_index('Kondisi'))

    with col_res2:
        if prediction == 1:
            # KASUS TSUNAMI (BAHAYA)
            st.error("⚠️ PERINGATAN DINI: BERPOTENSI TSUNAMI!")
            st.markdown(f"""
            **Analisis:**
            Model mendeteksi pola gempa yang memiliki kemiripan tinggi ({probability[1]*100:.2f}%) dengan kejadian Tsunami historis.
            
            **Rekomendasi:**
            1. Segera menjauh dari pantai.
            2. Cari tempat yang lebih tinggi.
            3. Pantau informasi resmi BMKG/Pemerintah.
            """)
             # Placeholder for diagram prompt

        else:
            # KASUS AMAN
            st.success("✅ STATUS: KEMUNGKINAN AMAN")
            st.markdown(f"""
            **Analisis:**
            Berdasarkan parameter yang dimasukkan (terutama kedalaman dan magnitude), model memprediksi kejadian ini **TIDAK** memicu Tsunami (Keyakinan: {probability[0]*100:.2f}%).
            
            **Catatan:**
            Tetap waspada terhadap gempa susulan.
            """)
