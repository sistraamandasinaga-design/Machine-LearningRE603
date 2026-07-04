# Image Classification using Transfer Learning
## Deteksi Jenis Sampah di Sekitar Kampus

**Nama:** Sistra Amanda Sinaga
**NIM:** 4222301011
**Kelas:** Pagi A
**Mata Kuliah:** RE603 — Practice dan Assignment 10

---

## 📋 Deskripsi Proyek

Proyek ini membangun model klasifikasi gambar untuk mendeteksi **6 jenis sampah** di sekitar kampus menggunakan teknik **Transfer Learning** dengan arsitektur **MobileNetV2** yang telah dilatih sebelumnya (pretrained) pada dataset ImageNet.

Selain membangun model, proyek ini juga melakukan:
1. Pengecekan apakah model mengalami **overfitting**
2. Penerapan **teknik penanganan overfitting** jika overfitting terdeteksi
3. Perbandingan performa model **sebelum vs sesudah** penanganan overfitting

---

## 📂 Dataset

- **Sumber:** [Trash Type Image Dataset — Kaggle](https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset)
- **Jumlah kelas:** 6 (cardboard, glass, metal, paper, plastic, trash)
- **Format:** 6 file ZIP terpisah, satu ZIP per kelas
- **Total gambar:** 2.527 gambar

| Kelas | Jumlah Gambar |
|---|---|
| Cardboard | 403 |
| Glass | 501 |
| Metal | 410 |
| Paper | 594 |
| Plastic | 482 |
| Trash | 137 |

> ⚠️ Dataset tidak seimbang (*imbalanced*) — kelas `trash` memiliki gambar paling sedikit, yang berdampak pada performa klasifikasi kelas tersebut.

**Split data:** 80% training (2.021 gambar) / 20% testing (506 gambar), dengan stratifikasi agar proporsi tiap kelas tetap seimbang di kedua set.

---

## 🧠 Metodologi

### 1. Preprocessing
- Semua gambar di-*resize* ke ukuran **128×128 piksel**
- Nilai piksel dijaga dalam skala **0–255** (bukan dinormalisasi manual), karena normalisasi ditangani oleh fungsi `preprocess_input` bawaan MobileNetV2 sesuai skema training ImageNet

### 2. Arsitektur Model — Transfer Learning
- **Base model:** MobileNetV2 (`weights='imagenet'`, `include_top=False`)
- Base model **dibekukan** (`trainable=False`) sehingga hanya bertindak sebagai *feature extractor*
- Ditambahkan layer klasifikasi baru: `GlobalAveragePooling2D → Dense → Dropout → Dense(softmax, 6 kelas)`

### 3. Tahap 1 — Model Baseline
Model baseline dilatih **tanpa** teknik anti-overfitting (tanpa augmentasi, dropout minimal, tanpa regularisasi) untuk membuktikan lewat data apakah overfitting benar-benar terjadi.

### 4. Tahap 2 — Deteksi Overfitting
Overfitting dicek dengan membandingkan **train accuracy vs validation accuracy** serta **train loss vs validation loss** melalui grafik dan perhitungan *gap*.

### 5. Tahap 3 — Penanganan Overfitting
Setelah overfitting terkonfirmasi, diterapkan lima teknik penanganan:
| Teknik | Fungsi |
|---|---|
| **Data Augmentation** (flip, rotate, zoom, contrast) | Menambah variasi data training agar model tidak menghafal |
| **Dropout** (0.3) | Mematikan sebagian neuron secara acak saat training |
| **L2 Regularization** | Memberi penalti pada bobot besar di layer Dense |
| **Early Stopping** | Menghentikan training saat val_loss tidak membaik |
| **ReduceLROnPlateau** | Menurunkan learning rate saat val_loss stagnan |

### 6. Tahap 4 — Fine-Tuning (Opsional)
20 layer terakhir MobileNetV2 dibuka kembali (`trainable=True`) dan dilatih dengan learning rate sangat kecil (1e-5) untuk penyesuaian lebih halus.

### 7. Evaluasi
Model dievaluasi pada data testing menggunakan accuracy, precision, F1-score, classification report, confusion matrix, dan akurasi per kelas.

---

## 📊 Hasil

### Perbandingan Baseline vs Setelah Penanganan Overfitting

| Metrik | Baseline | Setelah Handling |
|---|---|---|
| Train Accuracy | 99.78% | 87.90% |
| Validation Accuracy | 79.31% | 81.77% |
| **Gap (Train − Val)** | **20.47%** ⚠️ | **6.13%** ✅ |
| Best Validation Accuracy | 80.79% | 83.74% |

**Kesimpulan:** Model baseline mengalami **overfitting signifikan** (gap ±20%). Setelah penerapan data augmentation, dropout, L2 regularization, early stopping, dan ReduceLROnPlateau, gap menyusut menjadi ±6%, sementara akurasi validasi justru **meningkat**. Ini membuktikan teknik penanganan overfitting berhasil membuat model lebih general.

### Evaluasi Akhir pada Data Testing

- **Accuracy:** 83.99%
- **Precision (weighted):** 0.84
- **F1-score (weighted):** 0.84

| Kelas | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Cardboard | 0.84 | 0.86 | 0.85 | 81 |
| Glass | 0.91 | 0.82 | 0.86 | 100 |
| Metal | 0.78 | 0.89 | 0.83 | 82 |
| Paper | 0.84 | 0.84 | 0.84 | 119 |
| Plastic | 0.85 | 0.87 | 0.86 | 97 |
| Trash | 0.73 | 0.59 | 0.65 | 27 |

Kelas `glass` dan `plastic` memiliki performa terbaik, sedangkan kelas `trash` memiliki performa terendah — sejalan dengan jumlah datanya yang paling sedikit (137 gambar).

---

## 🛠️ Requirements

```
tensorflow
numpy>=1.26,<2
matplotlib
pillow
scikit-learn
pandas
seaborn
```

Semua dependency di-install otomatis lewat cell pertama di notebook.

---

## ▶️ Cara Menjalankan

1. **Buka Google Colab** ([colab.research.google.com](https://colab.research.google.com)) dan upload file `.ipynb` ini, atau buat notebook baru dan salin isinya.
2. **Aktifkan GPU:** menu `Runtime > Change runtime type > T4 GPU`.
3. **Upload dataset:** unggah 6 file ZIP (`cardboard.zip`, `glass.zip`, `metal.zip`, `paper.zip`, `plastic.zip`, `trash.zip`) ke Colab lewat panel **Files**, atau jalankan:
   ```python
   from google.colab import files
   uploaded = files.upload()
   ```
4. **Jalankan Cell 1** (install library) → setelah selesai, **restart runtime** (`Runtime > Restart runtime`) agar versi NumPy yang baru terpakai.
5. Setelah restart, **jalankan seluruh cell berurutan dari atas ke bawah** (mulai Cell 2, tidak perlu ulangi Cell 1).
6. Tunggu proses training baseline dan improved model selesai (± 15–25 menit total dengan GPU).
7. Lihat hasil akhir: grafik training, perbandingan baseline vs improved, confusion matrix, dan classification report di bagian akhir notebook.

---

## 📁 Struktur Notebook

1. Install & Import Library
2. Load Dataset dari ZIP
3. Train-Test Split
4. Visualisasi Sampel Data
5. Model Baseline (Transfer Learning MobileNetV2)
6. Training & Analisis Overfitting Baseline
7. Data Augmentation & Model Improved
8. Training Model Improved
9. Fine-Tuning (Opsional)
10. Perbandingan Baseline vs Improved
11. Evaluasi Akhir (Confusion Matrix, Classification Report)
12. Analisis Hasil & Kesimpulan

---

## 💡 Rekomendasi Pengembangan Lanjutan

- Menambah jumlah data untuk kelas `trash` yang masih sangat sedikit
- Mencoba base model transfer learning lain (EfficientNetB0, ResNet50) untuk dibandingkan
- Menerapkan `class_weight` saat training untuk mengatasi ketidakseimbangan kelas
- Menaikkan resolusi gambar (misalnya 160×160 atau 224×224) untuk hasil yang berpotensi lebih baik

---

## 📚 Referensi

- [Trash Type Image Dataset — Kaggle](https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset)
- [MobileNetV2 — Keras Applications Documentation](https://keras.io/api/applications/mobilenet/)
