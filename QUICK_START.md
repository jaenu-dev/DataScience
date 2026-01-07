# 🚀 Quick Start Guide

## Cara Cepat Menjalankan Aplikasi

### 1️⃣ Install Dependencies (Sekali saja)
```bash
pip install -r requirements.txt
```

### 2️⃣ Jalankan Streamlit App
```bash
streamlit run app_streamlit.py
```

### 3️⃣ Buka Browser
Aplikasi akan otomatis terbuka di: **http://localhost:8501**

---

## 📝 Cara Menggunakan

1. **Isi Form Input** - Masukkan data pasien (6 fitur)
2. **Klik Predict** - Tekan tombol prediksi
3. **Lihat Hasil** - Prediksi + probabilitas + interpretasi

---

## 🎯 Fitur Input

| Fitur | Deskripsi | Input Type |
|-------|-----------|------------|
| **cp** | Chest Pain Type (1-4) | Dropdown |
| **thalach** | Maximum Heart Rate (60-220) | Slider |
| **exang** | Exercise Induced Angina (Yes/No) | Dropdown |
| **oldpeak** | ST Depression (0.0-6.2) | Slider |
| **ca** | Number of Major Vessels (0-3) | Dropdown |
| **thal** | Thalassemia (Normal/Fixed/Reversible) | Dropdown |

---

## 📊 Output Prediksi

- ✅ **Prediksi**: DISEASE / NO DISEASE
- 📈 **Probabilitas**: Persentase untuk kedua class
- 💡 **Interpretasi**: Penjelasan hasil & rekomendasi

---

## ⚙️ Re-training Model (Optional)

Jika ingin training ulang:

```bash
# 1. Feature selection
python feature_selection.py

# 2. Train model
python train_model.py

# 3. Run app
streamlit run app_streamlit.py
```

---

## 📁 File Penting

- `app_streamlit.py` - Aplikasi utama
- `models/best_model.pkl` - Model trained
- `models/feature_names.pkl` - Fitur yang digunakan
- `README_DEPLOYMENT.md` - Dokumentasi lengkap

---

## ⚠️ Troubleshooting

**Error: ModuleNotFoundError**
```bash
pip install -r requirements.txt
```

**Error: Model not found**
```bash
python train_model.py
```

**Port sudah digunakan**
```bash
streamlit run app_streamlit.py --server.port 8502
```

---

**Ready to demo! 🎉**
