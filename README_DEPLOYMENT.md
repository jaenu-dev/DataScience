# Heart Disease Prediction - Deployment Guide

## 📋 Deskripsi Proyek

Aplikasi prediksi penyakit jantung menggunakan Machine Learning dengan deployment Streamlit.

**Dataset:** UCI Heart Disease (Cleveland)  
**Model:** Random Forest / Logistic Regression  
**Features:** 6 fitur terpilih (cp, thalach, exang, oldpeak, ca, thal)

## 🚀 Cara Menjalankan

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Feature Selection & Training

Jalankan script untuk analisis fitur dan training model:

```bash
# Analisis dan seleksi fitur
python feature_selection.py

# Training model dengan fitur terpilih
python train_model.py
```

Hasil:
- `selected_features.pkl` - Daftar fitur terpilih
- `models/best_model.pkl` - Model terbaik
- `models/feature_names.pkl` - Nama fitur
- `models/model_info.pkl` - Info model
- `models/scaler.pkl` - Scaler (jika diperlukan)

### 3. Jalankan Aplikasi Streamlit

```bash
streamlit run app_streamlit.py
```

Aplikasi akan terbuka di browser: `http://localhost:8501`

## 📊 Fitur yang Digunakan

1. **cp** - Chest Pain Type (1-4)
2. **thalach** - Maximum Heart Rate Achieved
3. **exang** - Exercise Induced Angina (0/1)
4. **oldpeak** - ST Depression
5. **ca** - Number of Major Vessels (0-3)
6. **thal** - Thalassemia (3/6/7)

## 📁 Struktur File

```
Tugas Besar FSD/
├── feature_selection.py      # Script analisis fitur
├── train_model.py             # Script training model
├── app_streamlit.py           # Aplikasi Streamlit
├── requirements.txt           # Dependencies
├── models/                    # Folder model
│   ├── best_model.pkl
│   ├── feature_names.pkl
│   ├── model_info.pkl
│   └── scaler.pkl (optional)
└── README_DEPLOYMENT.md       # File ini
```

## 🎯 Hasil Model

Model terbaik dipilih berdasarkan F1-Score (balance antara precision dan recall).

Metrics yang dievaluasi:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## ⚠️ Disclaimer

Aplikasi ini hanya untuk tujuan edukasi dan demonstrasi.  
Hasil prediksi tidak menggantikan diagnosis medis profesional.

## 👥 Tim

Tugas Besar - Fundamental Sains Data  
Semester 3

---

**Catatan:** Pastikan semua dependencies sudah terinstall sebelum menjalankan aplikasi.
