"""
Heart Disease Prediction App
Streamlit deployment untuk model prediksi penyakit jantung
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Page config
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    """Load model dan artifacts"""
    try:
        with open('models/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('models/feature_names.pkl', 'rb') as f:
            features = pickle.load(f)
        
        with open('models/model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        
        # Load scaler jika ada
        scaler = None
        if model_info.get('use_scaler', False):
            with open('models/scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
        
        return model, features, model_info, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Pastikan Anda sudah menjalankan `train_model.py` terlebih dahulu!")
        return None, None, None, None

# Load model
model, features, model_info, scaler = load_model()

# ==========================================
# HEADER
# ==========================================
st.title("❤️ Heart Disease Prediction")
st.markdown("### Prediksi Risiko Penyakit Jantung")
st.markdown("---")

if model is None:
    st.stop()

# ==========================================
# SIDEBAR - MODEL INFO
# ==========================================
with st.sidebar:
    st.header("ℹ️ Model Information")
    st.markdown(f"**Model:** {model_info['model_name']}")
    st.markdown(f"**Accuracy:** {model_info['accuracy']:.2%}")
    st.markdown(f"**Precision:** {model_info['precision']:.2%}")
    st.markdown(f"**Recall:** {model_info['recall']:.2%}")
    st.markdown(f"**F1-Score:** {model_info['f1_score']:.2%}")
    
    st.markdown("---")
    st.markdown("**Features Used:**")
    for feat in features:
        st.markdown(f"- {feat}")
    
    st.markdown("---")
    st.markdown("**Tugas Besar FSD**")
    st.markdown("Fundamental Sains Data")

# ==========================================
# MAIN CONTENT
# ==========================================

# Feature descriptions - BAHASA MUDAH DIPAHAMI
feature_info = {
    'cp': {
        'name': 'Tipe Nyeri Dada',
        'description': 'Jenis nyeri dada yang dirasakan saat beraktivitas',
        'help': '💡 Pilih yang paling sesuai dengan kondisi Anda',
        'options': {
            1: '😣 Nyeri dada khas (seperti diremas/ditekan saat aktivitas)',
            2: '😐 Nyeri dada tidak khas (nyeri tapi tidak seperti diremas)',
            3: '😕 Nyeri dada ringan (tidak berhubungan dengan jantung)',
            4: '😊 Tidak ada nyeri dada sama sekali'
        }
    },
    'thalach': {
        'name': 'Detak Jantung Maksimal',
        'description': 'Detak jantung tertinggi saat olahraga/aktivitas berat',
        'help': '💡 Normal: 150-170 bpm. Makin tua, makin rendah detak maksimal',
        'min': 60,
        'max': 220,
        'default': 150,
        'unit': 'bpm (detak per menit)'
    },
    'exang': {
        'name': 'Nyeri Dada Saat Olahraga',
        'description': 'Apakah merasa nyeri dada saat berolahraga atau aktivitas berat?',
        'help': '💡 Nyeri dada saat olahraga bisa jadi tanda masalah jantung',
        'options': {
            0: '✅ Tidak ada nyeri saat olahraga',
            1: '⚠️ Ya, ada nyeri saat olahraga'
        }
    },
    'oldpeak': {
        'name': 'Tingkat Kelelahan Jantung',
        'description': 'Seberapa lelah jantung saat olahraga (dari hasil EKG)',
        'help': '💡 0 = jantung tidak lelah, >2 = jantung cukup lelah. Nilai dari tes EKG',
        'min': 0.0,
        'max': 6.2,
        'default': 1.0,
        'step': 0.1,
        'labels': {
            0: '😊 Jantung fit',
            1: '😐 Normal',
            2: '😕 Agak lelah',
            3: '😣 Cukup lelah'
        }
    },
    'ca': {
        'name': 'Pembuluh Darah yang Tersumbat',
        'description': 'Jumlah pembuluh darah besar jantung yang tersumbat (dari tes)',
        'help': '💡 Hasil dari tes angiografi. 0 = tidak ada sumbatan, 3 = banyak sumbatan',
        'options': {
            0: '✅ Tidak ada pembuluh darah tersumbat (0)',
            1: '⚠️ 1 pembuluh darah tersumbat',
            2: '⚠️ 2 pembuluh darah tersumbat',
            3: '🚨 3 pembuluh darah tersumbat'
        }
    },
    'thal': {
        'name': 'Hasil Tes Darah Thalassemia',
        'description': 'Hasil tes darah untuk kelainan sel darah merah',
        'help': '💡 Thalassemia = kelainan darah turunan yang bisa pengaruhi jantung',
        'options': {
            3: '✅ Normal (tidak ada kelainan)',
            6: '⚠️ Ada kelainan permanen',
            7: '⚠️ Ada kelainan yang bisa diperbaiki'
        }
    }
}

# Create input form
st.header("📝 Input Data Pasien")
st.markdown("Masukkan data pasien untuk prediksi:")
st.info("💡 **Tips:** Jika tidak tahu nilai pasti, gunakan nilai default/perkiraan saja")

col1, col2 = st.columns(2)

input_data = {}

for idx, feat in enumerate(features):
    info = feature_info.get(feat, {})
    
    # Alternate between columns
    with col1 if idx % 2 == 0 else col2:
        st.subheader(info.get('name', feat))
        st.caption(info.get('description', ''))
        
        # Show help text if available
        if 'help' in info:
            st.markdown(f"_{info['help']}_")
        
        if 'options' in info:
            # Categorical feature
            selected = st.selectbox(
                f"Pilih {info.get('name', feat)}:",
                options=list(info['options'].keys()),
                format_func=lambda x: info['options'][x],
                key=feat
            )
            input_data[feat] = selected
        else:
            # Numerical feature
            label = f"Nilai {info.get('name', feat)}"
            if 'unit' in info:
                label += f" ({info['unit']})"
            
            value = st.slider(
                label,
                min_value=float(info.get('min', 0)),
                max_value=float(info.get('max', 100)),
                value=float(info.get('default', 50)),
                step=float(info.get('step', 1.0)),
                key=feat
            )
            input_data[feat] = value
            
            # Show label hints for oldpeak
            if 'labels' in info:
                for threshold, label_text in sorted(info['labels'].items()):
                    if value <= threshold:
                        st.caption(f"→ {label_text}")
                        break
                else:
                    st.caption(f"→ {list(info['labels'].values())[-1]}")

st.markdown("---")

# ==========================================
# PREDICTION
# ==========================================
if st.button("🔍 Predict", type="primary", use_container_width=True):
    # Prepare input
    input_df = pd.DataFrame([input_data])
    
    # Scale if needed
    if scaler is not None:
        input_scaled = scaler.transform(input_df)
    else:
        input_scaled = input_df
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]
    
    # Display results
    st.markdown("---")
    st.header("📊 Hasil Prediksi")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Prediksi",
            value="DISEASE" if prediction == 1 else "NO DISEASE",
            delta="Risiko Tinggi" if prediction == 1 else "Risiko Rendah",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            label="Probabilitas No Disease",
            value=f"{prediction_proba[0]:.1%}"
        )
    
    with col3:
        st.metric(
            label="Probabilitas Disease",
            value=f"{prediction_proba[1]:.1%}"
        )
    
    # Interpretation
    st.markdown("---")
    st.subheader("💡 Interpretasi")
    
    if prediction == 1:
        st.error("""
        **⚠️ Risiko Penyakit Jantung Terdeteksi**
        
        Berdasarkan data yang dimasukkan, model memprediksi adanya risiko penyakit jantung.
        
        **Rekomendasi:**
        - Segera konsultasi dengan dokter spesialis jantung
        - Lakukan pemeriksaan lebih lanjut
        - Jaga pola hidup sehat
        """)
    else:
        st.success("""
        **✅ Risiko Rendah**
        
        Berdasarkan data yang dimasukkan, model memprediksi risiko penyakit jantung rendah.
        
        **Tetap jaga kesehatan:**
        - Olahraga teratur
        - Pola makan sehat
        - Pemeriksaan rutin
        """)
    
    # Show input data
    with st.expander("📋 Lihat Data Input"):
        st.dataframe(input_df, use_container_width=True)

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>Disclaimer:</strong> Aplikasi ini hanya untuk tujuan edukasi dan demonstrasi. 
    Hasil prediksi tidak menggantikan diagnosis medis profesional.</p>
    <p>Tugas Besar - Fundamental Sains Data</p>
</div>
""", unsafe_allow_html=True)
