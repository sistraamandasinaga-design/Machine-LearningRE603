# ============================================================================
# NAÏVE BAYES CLASSIFICATION - PREDIKSI GAJI KARYAWAN
# Komponen UTS Machine Learning
# ============================================================================

# BAGIAN 1: TEORI DAN PENJELASAN
# ============================================================================
"""
TEORI NAÏVE BAYES CLASSIFICATION:

1. DEFINISI:
   Naïve Bayes adalah algoritma probabilistik yang berbasis pada Teorema Bayes.
   Algoritma ini mengasumsikan bahwa semua atribut/fitur bersifat independen
   secara kondisional terhadap target class.

2. FORMULA DASAR (TEOREMA BAYES):
   P(C|X) = [P(X|C) × P(C)] / P(X)
   
   Dimana:
   - P(C|X) = Posterior probability (probabilitas class C diberikan fitur X)
   - P(X|C) = Likelihood (probabilitas fitur X diberikan class C)
   - P(C) = Prior probability (probabilitas class C sebelum melihat data)
   - P(X) = Evidence (probabilitas fitur X di seluruh data)

3. ASUMSI NAÏVE:
   P(X|C) = P(X₁|C) × P(X₂|C) × ... × P(Xₙ|C)
   Setiap fitur diasumsikan independen satu sama lain.

4. KEUNTUNGAN:
   ✓ Cepat dan efisien
   ✓ Bekerja baik dengan data berdimensi tinggi
   ✓ Cocok untuk klasifikasi multi-class
   ✓ Membutuhkan data training yang relatif sedikit

5. KELEMAHAN:
   ✗ Asumsi independensi sering tidak berlaku
   ✗ Zero-frequency problem (jika ada fitur tidak muncul dalam class tertentu)
   ✗ Kurang akurat untuk dataset imbalanced

6. APLIKASI PADA KASUS INI:
   - Target: Memprediksi kategori gaji karyawan (Rendah/Sedang/Tinggi)
   - Fitur: Usia, Lama_Kerja, Departemen, Status_Karyawan
"""

# ============================================================================
# BAGIAN 2: IMPORT LIBRARY & KONFIGURASI
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score,
                             f1_score)
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi tampilan
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

print("="*80)
print("NAÏVE BAYES CLASSIFICATION - PREDIKSI GAJI KARYAWAN")
print("="*80)

# ============================================================================
# BAGIAN 3: LOADING & EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n[1] LOADING DATA...")
df = pd.read_csv('/mnt/user-data/uploads/dataset_karyawan_missing.csv')
print(f"✓ Dataset berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")

print("\n[2] INFORMASI DATASET:")
print(f"\nTipe Data:\n{df.dtypes}")
print(f"\nDimensi Dataset: {df.shape}")
print(f"\nBaris pertama:\n{df.head()}")

print("\n[3] STATISTIK DESKRIPTIF:")
print(df.describe())

print("\n[4] IDENTIFIKASI MISSING VALUES:")
missing_data = df.isnull().sum()
print(f"\n{missing_data[missing_data > 0]}")
print(f"\nPersentase Missing Values:\n{(missing_data[missing_data > 0] / len(df) * 100).round(2)}")

# ============================================================================
# BAGIAN 4: DATA PREPROCESSING
# ============================================================================
print("\n" + "="*80)
print("PREPROCESSING DATA")
print("="*80)

# 4.1 Duplikasi untuk keamanan
df_processed = df.copy()

print("\n[4.1] HANDLING MISSING VALUES:")

# Handling missing values pada Departemen (mode/nilai terbanyak)
print(f"  - Departemen: {df_processed['Departemen'].isnull().sum()} missing values")
mode_departemen = df_processed['Departemen'].mode()[0]
df_processed['Departemen'].fillna(mode_departemen, inplace=True)
print(f"    → Diisi dengan mode: {mode_departemen}")

# Handling missing values pada Gaji (median)
print(f"  - Gaji: {df_processed['Gaji'].isnull().sum()} missing values")
median_gaji = df_processed['Gaji'].median()
df_processed['Gaji'].fillna(median_gaji, inplace=True)
print(f"    → Diisi dengan median: Rp {median_gaji:,.0f}")

print(f"\n  ✓ Semua missing values berhasil ditangani")

# 4.2 Buat target variable dengan kategorisasi gaji
print("\n[4.2] KATEGORISASI GAJI (TARGET VARIABLE):")

# Menentukan batas kategori berdasarkan kuartil
Q1 = df_processed['Gaji'].quantile(0.33)
Q2 = df_processed['Gaji'].quantile(0.67)

def kategorikan_gaji(gaji):
    if gaji < Q1:
        return 'Rendah'
    elif gaji < Q2:
        return 'Sedang'
    else:
        return 'Tinggi'

df_processed['Kategori_Gaji'] = df_processed['Gaji'].apply(kategorikan_gaji)

print(f"\nBatas Kategori:")
print(f"  Rendah: < Rp {Q1:,.0f}")
print(f"  Sedang: Rp {Q1:,.0f} - Rp {Q2:,.0f}")
print(f"  Tinggi: > Rp {Q2:,.0f}")

print(f"\nDistribusi Kategori Gaji:")
print(df_processed['Kategori_Gaji'].value_counts())
print(df_processed['Kategori_Gaji'].value_counts(normalize=True).round(3))

# 4.3 Visualisasi distribusi
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Distribusi Gaji
axes[0, 0].hist(df_processed['Gaji'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].axvline(Q1, color='red', linestyle='--', label=f'Q1: Rp {Q1:,.0f}')
axes[0, 0].axvline(Q2, color='orange', linestyle='--', label=f'Q2: Rp {Q2:,.0f}')
axes[0, 0].set_xlabel('Gaji (Rp)', fontsize=11)
axes[0, 0].set_ylabel('Frekuensi', fontsize=11)
axes[0, 0].set_title('Distribusi Gaji Karyawan', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Pie chart kategori gaji
kategori_counts = df_processed['Kategori_Gaji'].value_counts()
colors = ['#ff9999', '#66b3ff', '#99ff99']
axes[0, 1].pie(kategori_counts, labels=kategori_counts.index, autopct='%1.1f%%',
               colors=colors, startangle=90)
axes[0, 1].set_title('Distribusi Kategori Gaji', fontweight='bold')

# Plot 3: Gaji berdasarkan Departemen
df_processed.boxplot(column='Gaji', by='Departemen', ax=axes[1, 0])
axes[1, 0].set_title('Distribusi Gaji per Departemen', fontweight='bold')
axes[1, 0].set_xlabel('Departemen')
axes[1, 0].set_ylabel('Gaji (Rp)')
plt.sca(axes[1, 0])
plt.xticks(rotation=45)

# Plot 4: Gaji berdasarkan Status
df_processed.boxplot(column='Gaji', by='Status_Karyawan', ax=axes[1, 1])
axes[1, 1].set_title('Distribusi Gaji per Status Karyawan', fontweight='bold')
axes[1, 1].set_xlabel('Status Karyawan')
axes[1, 1].set_ylabel('Gaji (Rp)')

plt.tight_layout()
plt.savefig('/home/claude/eda_gaji_karyawan.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualisasi EDA disimpan: eda_gaji_karyawan.png")
plt.close()

# 4.4 Persiapan fitur untuk modeling
print("\n[4.3] FEATURE ENGINEERING & ENCODING:")

# Pilih fitur yang akan digunakan
fitur_numerik = ['Usia', 'Lama_Kerja']
fitur_kategorik = ['Departemen', 'Status_Karyawan']

X = df_processed[fitur_numerik + fitur_kategorik].copy()
y = df_processed['Kategori_Gaji'].copy()

print(f"\nFitur yang digunakan:")
print(f"  Numerik: {fitur_numerik}")
print(f"  Kategorik: {fitur_kategorik}")

# Encoding variabel kategorik
label_encoders = {}
for col in fitur_kategorik:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
    print(f"\n  Encoding '{col}':")
    for i, label in enumerate(le.classes_):
        print(f"    {label} → {i}")

print(f"\nMatriks Fitur (X):")
print(X.head())

# ============================================================================
# BAGIAN 5: SPLIT DATA TRAINING & TESTING
# ============================================================================
print("\n" + "="*80)
print("SPLIT DATA TRAINING & TESTING")
print("="*80)

# Split dengan random state untuk reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining Set: {X_train.shape[0]} sampel ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Testing Set: {X_test.shape[0]} sampel ({X_test.shape[0]/len(X)*100:.1f}%)")

print(f"\nDistribusi Kelas di Training Set:")
print(y_train.value_counts())
print(f"\nDistribusi Kelas di Testing Set:")
print(y_test.value_counts())

# ============================================================================
# BAGIAN 6: TRAINING MODEL NAÏVE BAYES
# ============================================================================
print("\n" + "="*80)
print("TRAINING MODEL NAÏVE BAYES")
print("="*80)

print("\n[5.1] TRAINING GAUSSIAN NAÏVE BAYES:")
# Gaussian NB untuk data campuran (numerik + encoded kategorik)
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print("✓ Model Gaussian Naïve Bayes berhasil dilatih")

# Priors (probabilitas prior setiap kelas)
print(f"\nProbabilitas Prior (P(C)):")
for class_idx, class_name in enumerate(gnb.classes_):
    print(f"  P({class_name}) = {gnb.class_prior_[class_idx]:.4f}")

# Theta (mean dari setiap fitur per kelas)
print(f"\nMean Fitur per Kelas (θ):")
feature_names = fitur_numerik + fitur_kategorik
for class_idx, class_name in enumerate(gnb.classes_):
    print(f"\n  {class_name}:")
    for feat_idx, feat_name in enumerate(feature_names):
        print(f"    μ({feat_name}) = {gnb.theta_[class_idx, feat_idx]:.4f}")

# Variance (varians fitur per kelas)
print(f"\nVariance Fitur per Kelas (σ²):")
for class_idx, class_name in enumerate(gnb.classes_):
    print(f"\n  {class_name}:")
    for feat_idx, feat_name in enumerate(feature_names):
        print(f"    σ²({feat_name}) = {gnb.var_[class_idx, feat_idx]:.4f}")

# ============================================================================
# BAGIAN 7: PREDIKSI & EVALUASI MODEL
# ============================================================================
print("\n" + "="*80)
print("PREDIKSI & EVALUASI MODEL")
print("="*80)

print("\n[6.1] PREDIKSI PADA DATA TESTING:")
# Prediksi
y_pred = gnb.predict(X_test)
y_pred_proba = gnb.predict_proba(X_test)

print(f"✓ Prediksi berhasil dilakukan untuk {len(y_test)} sampel")

print(f"\n[6.2] ACCURACY SCORE:")
accuracy = accuracy_score(y_test, y_pred)
print(f"  Akurasi: {accuracy:.4f} ({accuracy*100:.2f}%)")

print(f"\n[6.3] CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, digits=4))

print(f"\n[6.4] DETAILED METRICS:")
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"  Precision (Weighted): {precision:.4f}")
print(f"  Recall (Weighted): {recall:.4f}")
print(f"  F1-Score (Weighted): {f1:.4f}")

# ============================================================================
# BAGIAN 8: CONFUSION MATRIX & VISUALISASI
# ============================================================================
print("\n[6.5] CONFUSION MATRIX:")
cm = confusion_matrix(y_test, y_pred, labels=gnb.classes_)
print(cm)

# Visualisasi confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=gnb.classes_, yticklabels=gnb.classes_,
            ax=axes[0], cbar_kws={'label': 'Count'})
axes[0].set_title('Confusion Matrix - Naïve Bayes', fontweight='bold', fontsize=12)
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

# Performance Metrics
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metrics_values = [accuracy, precision, recall, f1]
colors_bar = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

bars = axes[1].bar(metrics_names, metrics_values, color=colors_bar, alpha=0.8, edgecolor='black')
axes[1].set_ylim([0, 1])
axes[1].set_title('Performance Metrics', fontweight='bold', fontsize=12)
axes[1].set_ylabel('Score')
axes[1].grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, metrics_values):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}',
                ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('/home/claude/model_evaluation.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualisasi evaluasi model disimpan: model_evaluation.png")
plt.close()

# ============================================================================
# BAGIAN 9: PREDICTION PROBABILITY ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("ANALISIS PROBABILITY PREDIKSI")
print("="*80)

print("\n[7.1] SAMPLE PREDICTIONS DENGAN PROBABILITY:")
print("\nContoh 10 prediksi pertama:")
print(f"{'No':<4} {'Usia':<6} {'Lama_K':<8} {'Dept':<10} {'Status':<8} {'True':<8} {'Pred':<8} {'Confidence':<12}")
print("-" * 70)

for i in range(min(10, len(X_test))):
    usia = X_test.iloc[i]['Usia']
    lama_k = X_test.iloc[i]['Lama_Kerja']
    
    # Get department and status labels
    dept_idx = int(X_test.iloc[i]['Departemen'])
    dept_label = label_encoders['Departemen'].inverse_transform([dept_idx])[0]
    
    status_idx = int(X_test.iloc[i]['Status_Karyawan'])
    status_label = label_encoders['Status_Karyawan'].inverse_transform([status_idx])[0]
    
    true_class = y_test.iloc[i]
    pred_class = y_pred[i]
    confidence = np.max(y_pred_proba[i])
    
    print(f"{i+1:<4} {usia:<6.0f} {lama_k:<8.0f} {dept_label:<10} {status_label:<8} {true_class:<8} {pred_class:<8} {confidence:<12.4f}")

# ============================================================================
# BAGIAN 10: PREDICTION UNTUK DATA BARU
# ============================================================================
print("\n" + "="*80)
print("PREDIKSI UNTUK DATA KARYAWAN BARU")
print("="*80)

print("\n[8.1] CONTOH PREDIKSI DATA BARU:")

# Data baru untuk diprediksi
new_employees = {
    'Usia': [28, 45, 52],
    'Lama_Kerja': [2, 15, 20],
    'Departemen': ['HRD', 'Produksi', 'Logistik'],
    'Status_Karyawan': ['Kontrak', 'Tetap', 'Tetap']
}

df_new = pd.DataFrame(new_employees)

# Encoding data baru
df_new_encoded = df_new.copy()
df_new_encoded['Departemen'] = label_encoders['Departemen'].transform(df_new_encoded['Departemen'])
df_new_encoded['Status_Karyawan'] = label_encoders['Status_Karyawan'].transform(df_new_encoded['Status_Karyawan'])

# Prediksi
predictions_new = gnb.predict(df_new_encoded)
probabilities_new = gnb.predict_proba(df_new_encoded)

print("\nHasil Prediksi untuk Karyawan Baru:")
print("-" * 100)
for i in range(len(df_new)):
    print(f"\nKaryawan {i+1}:")
    print(f"  Usia: {df_new.iloc[i]['Usia']} tahun")
    print(f"  Lama Kerja: {df_new.iloc[i]['Lama_Kerja']} tahun")
    print(f"  Departemen: {df_new.iloc[i]['Departemen']}")
    print(f"  Status: {df_new.iloc[i]['Status_Karyawan']}")
    print(f"  Prediksi Kategori Gaji: {predictions_new[i]}")
    print(f"  Confidence:")
    for j, class_name in enumerate(gnb.classes_):
        print(f"    P({class_name}) = {probabilities_new[i][j]:.4f}")

# ============================================================================
# BAGIAN 11: FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Hitung correlation dengan target (setelah encode)
X_with_target = X.copy()
y_encoded = LabelEncoder().fit_transform(y)
X_with_target['Target'] = y_encoded

correlations = X_with_target.corr()['Target'].drop('Target').abs().sort_values(ascending=False)

print("\nKorelasi Fitur dengan Target:")
for feature, corr in correlations.items():
    print(f"  {feature}: {corr:.4f}")

# Visualisasi
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Feature Importance
axes[0].barh(correlations.index, correlations.values, color='steelblue', alpha=0.8, edgecolor='black')
axes[0].set_xlabel('Absolute Correlation', fontsize=11)
axes[0].set_title('Feature Importance (Correlation with Target)', fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)

# Prediction Distribution
pred_distribution = pd.Series(y_pred).value_counts().sort_index()
axes[1].bar(pred_distribution.index, pred_distribution.values, 
           color=['#ff9999', '#66b3ff', '#99ff99'], alpha=0.8, edgecolor='black')
axes[1].set_xlabel('Kategori Gaji', fontsize=11)
axes[1].set_ylabel('Jumlah Prediksi', fontsize=11)
axes[1].set_title('Distribusi Hasil Prediksi pada Test Set', fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

for i, v in enumerate(pred_distribution.values):
    axes[1].text(i, v + 0.3, str(v), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('/home/claude/feature_importance.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualisasi feature importance disimpan: feature_importance.png")
plt.close()

# ============================================================================
# BAGIAN 12: SUMMARY & KESIMPULAN
# ============================================================================
print("\n" + "="*80)
print("SUMMARY & KESIMPULAN")
print("="*80)

summary_text = f"""
RINGKASAN HASIL NAÏVE BAYES CLASSIFICATION:

1. DATASET:
   - Total sampel: {len(df)}
   - Training set: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)
   - Testing set: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)
   - Fitur numerik: {len(fitur_numerik)}
   - Fitur kategorik: {len(fitur_kategorik)}
   - Target classes: {len(gnb.classes_)} (Rendah, Sedang, Tinggi)

2. PREPROCESSING:
   - Missing values pada Departemen: {len(df) - df['Departemen'].notna().sum()} → diisi dengan mode
   - Missing values pada Gaji: {len(df) - df['Gaji'].notna().sum()} → diisi dengan median
   - Target variable: Kategori Gaji (binning berbasis kuartil)

3. MODEL PERFORMANCE:
   - Algoritma: Gaussian Naïve Bayes
   - Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)
   - Precision: {precision:.4f}
   - Recall: {recall:.4f}
   - F1-Score: {f1:.4f}

4. CONFUSION MATRIX:
   Prediksi benar untuk masing-masing kelas:
   - Rendah: {cm[0, 0]} / {cm[0].sum()} ({cm[0, 0]/cm[0].sum()*100:.1f}%)
   - Sedang: {cm[1, 1]} / {cm[1].sum()} ({cm[1, 1]/cm[1].sum()*100:.1f}%)
   - Tinggi: {cm[2, 2]} / {cm[2].sum()} ({cm[2, 2]/cm[2].sum()*100:.1f}%)

5. INTERPRETASI MODEL:
   Model Naïve Bayes mampu memprediksi kategori gaji karyawan dengan akurasi
   {accuracy*100:.2f}%. Fitur yang paling berpengaruh adalah {correlations.index[0]}
   dengan korelasi {correlations.values[0]:.4f}.

6. KEPUTUSAN BISNIS:
   ✓ Model dapat digunakan untuk prediksi awal gaji karyawan baru
   ✓ Perlu validasi lebih lanjut dengan data actual untuk optimasi
   ✓ Rekomendasi: Tambahkan fitur lain seperti tingkat pendidikan, sertifikasi

FILES OUTPUT:
   - eda_gaji_karyawan.png: Visualisasi exploratory data analysis
   - model_evaluation.png: Confusion matrix dan performance metrics
   - feature_importance.png: Feature importance analysis
"""

print(summary_text)

# Simpan summary ke file
with open('/home/claude/summary_naivebayes.txt', 'w', encoding='utf-8') as f:
    f.write(summary_text)
print("\n✓ Summary disimpan: summary_naivebayes.txt")

print("\n" + "="*80)
print("ANALISIS NAÏVE BAYES CLASSIFICATION SELESAI!")
print("="*80)
