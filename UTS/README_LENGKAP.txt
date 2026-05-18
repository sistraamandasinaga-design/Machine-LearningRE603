═══════════════════════════════════════════════════════════════════════════════
    NAÏVE BAYES CLASSIFICATION - PREDIKSI GAJI KARYAWAN PT XYZ
    UTS Machine Learning - Polibatam
═══════════════════════════════════════════════════════════════════════════════

📋 DAFTAR FILE YANG DISEDIAKAN:
═══════════════════════════════════════════════════════════════════════════════

1. Naive_Bayes_UTS_Machine_Learning.ipynb
   ✓ Jupyter Notebook lengkap dengan teori dan praktik
   ✓ Format resmi untuk UTS Machine Learning
   ✓ Terdiri dari:
     - Penjelasan teoritis tentang Naïve Bayes
     - Langkah-langkah coding praktek
     - Analisis hasil dan visualisasi
   ✓ Dapat dijalankan di Jupyter Notebook atau Google Colab

2. naive_bayes_karyawan.py
   ✓ Script Python standalone yang dapat dijalankan langsung
   ✓ Berisi implementasi lengkap Naïve Bayes Classification
   ✓ Output otomatis ke file visualisasi dan summary
   ✓ Berguna untuk testing atau batch processing

3. eda_gaji_karyawan.png
   ✓ Visualisasi Exploratory Data Analysis
   ✓ Menampilkan:
     - Distribusi gaji karyawan
     - Pie chart kategori gaji
     - Box plot gaji per departemen
     - Scatter plot Usia vs Gaji
   ✓ Ukuran: 308 KB (High resolution 300 DPI)

4. model_evaluation.png
   ✓ Visualisasi evaluasi model
   ✓ Menampilkan:
     - Confusion Matrix sebagai heatmap
     - Performance metrics (Accuracy, Precision, Recall, F1-Score)
   ✓ Ukuran: 144 KB

5. feature_importance.png
   ✓ Analisis pentingnya fitur
   ✓ Menampilkan:
     - Feature importance (berdasarkan korelasi)
     - Distribusi hasil prediksi
   ✓ Ukuran: 138 KB

6. summary_naivebayes.txt
   ✓ Ringkasan lengkap hasil analisis
   ✓ Berisi:
     - Dataset statistics
     - Model performance metrics
     - Confusion matrix analysis
     - Business recommendations

7. README.md (File ini)
   ✓ Panduan lengkap penggunaan dan penjelasan


═══════════════════════════════════════════════════════════════════════════════
🚀 PANDUAN PENGGUNAAN
═══════════════════════════════════════════════════════════════════════════════

### OPSI 1: MENGGUNAKAN JUPYTER NOTEBOOK (REKOMENDASI UNTUK UTS)

1. Buka file: Naive_Bayes_UTS_Machine_Learning.ipynb
2. Gunakan aplikasi:
   - Jupyter Notebook (lokal)
   - Google Colab
   - VS Code dengan extension Jupyter
   - Anaconda Navigator

3. Jalankan setiap cell secara berurutan
4. Baca penjelasan teori di setiap markdown cell
5. Amati output dari setiap code cell

### OPSI 2: MENJALANKAN SCRIPT PYTHON

1. Pastikan dataset 'dataset_karyawan_missing.csv' ada di directory yang sama
2. Install requirements:
   pip install pandas numpy scikit-learn matplotlib seaborn

3. Jalankan script:
   python naive_bayes_karyawan.py

4. Semua visualisasi akan disimpan otomatis di folder output


═══════════════════════════════════════════════════════════════════════════════
📚 TEORI NAÏVE BAYES (RINGKAS)
═══════════════════════════════════════════════════════════════════════════════

DEFINISI:
Naïve Bayes adalah algoritma klasifikasi probabilistik berbasis Teorema Bayes
yang mengasumsikan semua fitur bersifat independen secara kondisional.

FORMULA DASAR:
P(C|X) = [P(X|C) × P(C)] / P(X)

Dimana:
- P(C|X) = Posterior (probabilitas class C diberikan fitur X)
- P(X|C) = Likelihood (probabilitas fitur X diberikan class C)
- P(C)   = Prior (probabilitas class C)
- P(X)   = Evidence (probabilitas fitur X)

ASUMSI NAÏVE:
P(X|C) = P(X₁|C) × P(X₂|C) × ... × P(Xₙ|C)

LANGKAH KLASIFIKASI:
1. Hitung prior probability setiap kelas
2. Hitung likelihood setiap fitur
3. Hitung posterior probability menggunakan Bayes theorem
4. Pilih class dengan posterior tertinggi

KELEBIHAN:
✓ Cepat dan efisien (O(n))
✓ Baik untuk high-dimensional data
✓ Cocok untuk real-time prediction
✓ Membutuhkan data training sedikit

KELEMAHAN:
✗ Asumsi independensi sering tidak berlaku
✗ Zero-frequency problem
✗ Kurang akurat untuk imbalanced data
✗ Probability estimate kurang akurat


═══════════════════════════════════════════════════════════════════════════════
📊 DESKRIPSI DATASET
═══════════════════════════════════════════════════════════════════════════════

NAMA DATASET: dataset_karyawan_missing.csv

DIMENSI:
- Total sampel: 200 karyawan
- Fitur: 7 kolom (ID, Nama, Departemen, Gaji, Usia, Lama_Kerja, Status_Karyawan)

MISSING VALUES:
- Departemen: 10 missing (5%) → Diisi dengan mode (HRD)
- Gaji: 10 missing (5%) → Diisi dengan median (Rp 5,040,414)

FITUR YANG DIGUNAKAN:
1. Usia (numerik) - Range: 22-59 tahun
2. Lama_Kerja (numerik) - Range: 0-29 tahun
3. Departemen (kategorik) - 5 kategori: HRD, Logistik, Produksi, QC, Missing
4. Status_Karyawan (kategorik) - 2 kategori: Tetap, Kontrak

TARGET VARIABLE:
Kategori_Gaji (dibuat dari binning Gaji):
- Rendah: < Rp 4,769,188 (31.5% sampel)
- Sedang: Rp 4,769,188 - Rp 5,379,417 (32.0% sampel)
- Tinggi: > Rp 5,379,417 (36.5% sampel)


═══════════════════════════════════════════════════════════════════════════════
🔍 HASIL ANALISIS
═══════════════════════════════════════════════════════════════════════════════

TRAIN-TEST SPLIT:
- Training set: 160 sampel (80%)
- Testing set: 40 sampel (20%)
- Stratified untuk mempertahankan proporsi kelas

MODEL PERFORMANCE:
- Algorithm: Gaussian Naïve Bayes
- Accuracy: 32.50%
- Precision (weighted): 0.3367
- Recall (weighted): 0.3250
- F1-Score (weighted): 0.3139

CONFUSION MATRIX:
                Predicted Rendah  Sedang  Tinggi
True Rendah                 3      3       7
True Sedang                 1      3       9
True Tinggi                 4      3       7

Per-class accuracy:
- Rendah: 23.1% (3/13)
- Sedang: 23.1% (3/13)
- Tinggi: 50.0% (7/14)

FEATURE IMPORTANCE:
1. Lama_Kerja: 0.0864 (most important)
2. Departemen: 0.0231
3. Usia: 0.0208
4. Status_Karyawan: 0.0103

MODEL PARAMETERS (LEARNED FROM DATA):
Prior Probabilities:
- P(Rendah) = 0.3125
- P(Sedang) = 0.3187
- P(Tinggi) = 0.3688

Mean values per class tersimpan di model.theta_
Variance per class tersimpan di model.var_


═══════════════════════════════════════════════════════════════════════════════
💡 CONTOH PREDIKSI
═══════════════════════════════════════════════════════════════════════════════

Prediksi untuk karyawan baru:
────────────────────────────────────────────────────────────────────────────────

Karyawan 1:
  Usia: 28 tahun
  Lama Kerja: 2 tahun
  Departemen: HRD
  Status: Kontrak
  → Prediksi: TINGGI
  → P(Rendah) = 0.2181, P(Sedang) = 0.3815, P(Tinggi) = 0.4004

Karyawan 2:
  Usia: 45 tahun
  Lama Kerja: 15 tahun
  Departemen: Produksi
  Status: Tetap
  → Prediksi: SEDANG
  → P(Rendah) = 0.2986, P(Sedang) = 0.3738, P(Tinggi) = 0.3276

Karyawan 3:
  Usia: 52 tahun
  Lama Kerja: 20 tahun
  Departemen: Logistik
  Status: Tetap
  → Prediksi: TINGGI
  → P(Rendah) = 0.3164, P(Sedang) = 0.3238, P(Tinggi) = 0.3598


═══════════════════════════════════════════════════════════════════════════════
⚠️ POIN-POIN PENTING
═══════════════════════════════════════════════════════════════════════════════

1. AKURASI MODEL:
   - Akurasi 32.50% menunjukkan model masih perlu improvement
   - Kemungkinan penyebab:
     * Fitur yang tidak cukup informatif
     * Asumsi independensi tidak berlaku
     * Data yang kurang banyak
     * Imbalance dalam feature distribution

2. CONFIDENCE SCORE:
   - Probability dari prediksi menunjukkan confidence level
   - Score tinggi (>0.5) berarti model confident
   - Score rendah (<0.4) berarti ada ambiguity

3. CONFUSION MATRIX INTERPRETATION:
   - Model terbaik dalam memprediksi kategori "Tinggi" (50% recall)
   - Model kurang baik untuk "Rendah" dan "Sedang" (~23% recall)
   - Saran: Tambahkan lebih banyak fitur atau gunakan algoritma lain

4. CLASS BALANCE:
   - Dataset sudah well-balanced (31-36% per class)
   - Tidak perlu SMOTE oversampling

5. FEATURE IMPORTANCE:
   - Lama_Kerja adalah fitur paling penting
   - Korelasi keseluruhan fitur rendah → perlu feature engineering


═══════════════════════════════════════════════════════════════════════════════
🎯 REKOMENDASI UNTUK IMPROVEMENT
═══════════════════════════════════════════════════════════════════════════════

1. TAMBAH FITUR BARU:
   - Tingkat Pendidikan (SMA/D3/S1/S2)
   - Sertifikasi Profesional
   - Pengalaman Industri
   - Performance Rating
   - Divisi/Bagian
   - Shift/Jadwal Kerja

2. OPTIMIZE MODEL:
   - Coba algoritma lain:
     * Random Forest (ensemble method)
     * Gradient Boosting (XGBoost, LightGBM)
     * SVM dengan kernel
     * Neural Networks
   
   - Hyperparameter tuning:
     * Grid Search / Random Search
     * Cross-validation untuk robust evaluation
   
   - Feature scaling/normalization untuk algoritma yang sensitif

3. HANDLE CLASS IMBALANCE:
   - SMOTE oversampling jika diperlukan
   - Class weights dalam loss function
   - Stratified k-fold cross-validation

4. FEATURE ENGINEERING:
   - Interaction terms (Usia × Lama_Kerja)
   - Polynomial features
   - Domain-specific features
   - Feature selection (Chi-square, mutual information)

5. DATA QUALITY:
   - Collect lebih banyak data training
   - Handle outliers lebih baik
   - Domain expertise untuk feature creation


═══════════════════════════════════════════════════════════════════════════════
📖 CARA PRESENTASI UTS
═══════════════════════════════════════════════════════════════════════════════

Slide 1: JUDUL DAN PROBLEM STATEMENT
- Naïve Bayes Classification untuk Prediksi Gaji Karyawan
- Konteks: PT XYZ ingin sistem objektif untuk menentukan gaji karyawan baru

Slide 2: TEORI NAÏVE BAYES
- Penjelasan singkat rumus Bayes
- Asumsi independensi
- Kapan menggunakan Naïve Bayes

Slide 3: DATA & PREPROCESSING
- Deskripsi dataset
- Missing value handling
- Target variable creation (kategorisasi gaji)
- Feature selection

Slide 4: MODEL BUILDING
- Train-test split strategi
- Hyperparameter (jika ada)
- Training process

Slide 5: RESULTS & EVALUATION
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix interpretation
- Feature importance analysis

Slide 6: PREDICTIONS EXAMPLES
- Contoh prediksi untuk data baru
- Probability analysis
- Business implications

Slide 7: KESIMPULAN & REKOMENDASI
- Ringkasan findings
- Limitations
- Future improvements
- Business recommendations


═══════════════════════════════════════════════════════════════════════════════
❓ FAQ DAN TROUBLESHOOTING
═══════════════════════════════════════════════════════════════════════════════

Q: Mengapa akurasi rendah (32.5%)?
A: Beberapa kemungkinan:
   - Fitur tidak cukup informatif untuk prediksi
   - Asumsi independensi tidak berlaku
   - Relationship antara features dan target complex
   - Coba algoritma lain (Random Forest, SVM)

Q: Bagaimana cara meningkatkan akurasi?
A: 
   1. Tambah fitur yang lebih informatif
   2. Coba algoritma lain
   3. Feature engineering
   4. Hyperparameter tuning
   5. Collect lebih banyak data

Q: Apa itu probability dalam prediksi?
A: Confidence level dari model bahwa sample termasuk class tertentu.
   Nilai berkisar 0-1. Semakin tinggi semakin confident.

Q: Berapa jumlah data minimal untuk Naïve Bayes?
A: Tergantung kompleksitas problem. Umumnya 100-1000 samples sudah cukup.
   Dataset ini (200 samples) masih reasonable.

Q: Bagaimana interpretasi Confusion Matrix?
A: 
   - Diagonal (top-left to bottom-right) = prediksi benar
   - Off-diagonal = prediksi salah
   - Row = actual class
   - Column = predicted class

Q: Bisa digunakan di production?
A: Dengan akurasi 32.5%, sangat tidak disarankan untuk production.
   Perlu improvement atau validation dengan domain experts dulu.


═══════════════════════════════════════════════════════════════════════════════
📞 REFERENSI DAN RESOURCES
═══════════════════════════════════════════════════════════════════════════════

Textbooks:
- Andrew Ng - Machine Learning Course (Coursera)
- "Machine Learning" by Tom Mitchell
- "Hands-On Machine Learning" by Aurélien Géron

Documentation:
- Scikit-learn Naive Bayes: https://scikit-learn.org/stable/modules/naive_bayes.html
- Wikipedia Naive Bayes: https://en.wikipedia.org/wiki/Naive_Bayes_classifier

GitHub Repositories:
- Scikit-learn: https://github.com/scikit-learn/scikit-learn
- TensorFlow: https://github.com/tensorflow/tensorflow

Polibatam Resources:
- Learning Platform: https://learning.polibatam.ac.id/
- Machine Learning Course: Sesuai course code


═══════════════════════════════════════════════════════════════════════════════
✅ CHECKLIST UNTUK SUBMISSION UTS
═══════════════════════════════════════════════════════════════════════════════

Teori:
☑ Penjelasan Teorema Bayes
☑ Asumsi Naïve Bayes
☑ Formula dan rumus matematis
☑ Perbedaan dengan algoritma lain
☑ Kelebihan dan kelemahan

Praktik (Coding):
☑ Data loading dan exploration
☑ Missing value handling
☑ Feature engineering
☑ Data preprocessing
☑ Model training
☑ Prediction dan evaluation
☑ Visualization

Dokumentasi:
☑ Code comments yang jelas
☑ Penjelasan setiap tahap
☑ Output visualisasi
☑ Hasil numerik (metrics)
☑ Interpretasi hasil

Submission Format:
☑ Jupyter Notebook (.ipynb)
☑ Python Script (.py) - Optional
☑ Dokumentasi README
☑ Visualisasi (PNG)
☑ Summary Report


═══════════════════════════════════════════════════════════════════════════════
🎓 KESIMPULAN
═══════════════════════════════════════════════════════════════════════════════

Naïve Bayes Classification adalah algoritma klasifikasi yang powerful dan
efficient untuk berbagai problem praktis. Meskipun asumsi independensinya
sering tidak berlaku di dunia nyata, algoritma ini tetap memberikan hasil
yang reasonable, terutama untuk rapid prototyping dan baseline models.

Dalam kasus prediksi gaji karyawan PT XYZ ini, Naïve Bayes telah
mendemonstrasikan bagaimana probabilistic approach dapat digunakan untuk
klasifikasi. Dengan improvement pada feature engineering dan potentially
algoritma lain, akurasi dapat ditingkatkan lebih lanjut.

Key takeaways:
1. Understanding teori probabilitas penting dalam machine learning
2. Data preprocessing menentukan kualitas model
3. Evaluation metrics memberikan insight yang berbeda
4. Iterasi dan improvement adalah bagian dari ML development
5. Dokumentasi dan interpretasi hasil sama pentingnya dengan akurasi


═══════════════════════════════════════════════════════════════════════════════
Dibuat: 4 Mei 2026
Versi: 1.0
Status: Ready for UTS Submission
═══════════════════════════════════════════════════════════════════════════════
