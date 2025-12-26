# Reduksi Dimensi Wholesale Customers Dataset

## Anggota Kelompok

- Janu Farras Saguna
- Farid Faizal Hakim

## Pendahuluan

Proyek ini bertujuan untuk menerapkan teknik reduksi dimensi pada _Wholesale Customers Dataset_
menggunakan Principal Component Analysis (PCA) dan t-Distributed Stochastic Neighbor Embedding (t-SNE).
Reduksi dimensi dilakukan untuk mempermudah visualisasi data berdimensi tinggi serta
mengidentifikasi pola dan segmentasi pelanggan berdasarkan saluran distribusi (_Channel_).

## Tujuan Proyek

1. Menerapkan PCA dan t-SNE untuk mereduksi dimensi data pelanggan grosir.
2. Memvisualisasikan hasil reduksi dimensi dalam ruang dua dimensi.
3. Membandingkan hasil dan karakteristik PCA dan t-SNE.
4. Menentukan metode yang paling sesuai untuk visualisasi segmentasi pelanggan.

## Dataset

Dataset yang digunakan adalah **Wholesale Customers Dataset** dari UCI Machine Learning Repository.
Dataset ini terdiri dari 440 data pelanggan dengan 8 fitur utama, yaitu:

- **Channel**: Saluran distribusi (1 = Horeca, 2 = Retail)
- **Region**: Wilayah geografis (1 = Lisbon, 2 = Oporto, 3 = Other)
- **Fresh**: Pengeluaran tahunan untuk produk segar
- **Milk**: Pengeluaran tahunan untuk produk susu
- **Grocery**: Pengeluaran tahunan untuk produk grosir
- **Frozen**: Pengeluaran tahunan untuk produk beku
- **Detergents_Paper**: Pengeluaran tahunan untuk deterjen dan produk kertas
- **Delicassen**: Pengeluaran tahunan untuk produk delikatesen

## Metode

### 1. Preprocessing Data

Kolom **Channel** digunakan sebagai label untuk visualisasi, sedangkan kolom **Region**
tidak disertakan dalam proses reduksi dimensi karena bersifat kategorikal.
Fitur-fitur numerik pengeluaran distandardisasi menggunakan **StandardScaler**
agar setiap fitur memiliki skala yang sama (rata-rata 0 dan standar deviasi 1).
Standardisasi ini penting karena PCA dan t-SNE sensitif terhadap perbedaan skala data.

### 2. Principal Component Analysis (PCA)

PCA diterapkan untuk mereduksi data menjadi dua komponen utama (PC1 dan PC2).
Hasil PCA menunjukkan bahwa:

- **PC1 menjelaskan sekitar 44% variasi data**
- **PC2 menjelaskan sekitar 28% variasi data**
- **Total explained variance ≈ 72%**

Visualisasi PCA menunjukkan adanya pemisahan antara pelanggan Horeca dan Retail,
namun masih terdapat tumpang tindih karena PCA bersifat linier.

### 3. t-Distributed Stochastic Neighbor Embedding (t-SNE)

t-SNE diterapkan untuk mereduksi data menjadi dua dimensi dengan fokus pada pelestarian
struktur lokal data. Visualisasi t-SNE menunjukkan pemisahan cluster yang lebih jelas
antara pelanggan Horeca dan Retail dibandingkan PCA, dengan cluster yang lebih rapat
dan terdefinisi dengan baik.

## Hasil dan Pembahasan

- **PCA** efektif dalam merangkum variasi global data dan memberikan informasi kuantitatif
  melalui explained variance, namun kurang optimal dalam memisahkan cluster secara visual.
- **t-SNE** menghasilkan visualisasi yang lebih jelas dalam mengungkap struktur cluster,
  sehingga lebih cocok untuk eksplorasi dan visualisasi segmentasi pelanggan.

## Kesimpulan

Berdasarkan hasil analisis, **t-SNE lebih sesuai untuk tujuan visualisasi segmentasi pelanggan**
pada Wholesale Customers Dataset karena mampu menampilkan pemisahan cluster yang lebih jelas.
Sementara itu, PCA tetap berguna sebagai metode reduksi dimensi linier dan
sebagai tahap awal eksplorasi data.

## Rekomendasi

Sebagai pengembangan lanjutan, hasil reduksi dimensi ini dapat digunakan
sebagai dasar untuk penerapan algoritma clustering seperti K-Means
atau sebagai pendukung analisis strategi pemasaran yang lebih terarah
untuk pelanggan Horeca dan Retail.
