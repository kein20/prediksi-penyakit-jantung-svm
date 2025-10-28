"""
Project Akhir Kecerdasan Buatan:
Klasifikasi Penyakit Jantung menggunakan Support Vector Machine (SVM)
"""

# 1. Mengimpor pustaka yang diperlukan
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 2. Memuat Dataset
# Membaca data dari file CSV yang ada di dalam repository
df = pd.read_csv("heart_disease_data.csv")

# 3. Data Understanding
print("--- Informasi Awal Dataset ---")
print("Bentuk Data (Baris, Kolom): ", df.shape)
print("\nInformasi Tipe Data:")
df.info()
print("\nStatistik Deskriptif:")
print(df.describe())
print("\nJumlah Pasien (Target):")
print(df.target.value_counts())

# 4. Cleaning Data
print("\n--- Proses Pembersihan Data ---")
print("Jumlah nilai null sebelum dibersihkan:", df.isnull().sum().sum())
print("Jumlah data duplikat:", df.duplicated().sum())
df = df.drop_duplicates()
print("Bentuk Data setelah menghapus duplikat:", df.shape)

# 5. Exploratory Data Analysis (EDA)
# Visualisasi distribusi umur
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='age', bins=10, kde=True, color='#6A1E55')
plt.title('Distribusi Umur Pasien')
plt.xlabel('Umur')
plt.ylabel('Frekuensi')
plt.show()

# Visualisasi proporsi penyakit jantung
plt.figure(figsize=(8, 6))
df['target'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#4b7bec', '#fc5c65'])
plt.title('Proporsi Diagnosis Penyakit Jantung (1) vs Sehat (0)')
plt.ylabel('')
plt.show()

# Visualisasi korelasi antar fitur
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), cmap='YlOrRd', annot=False)
plt.title('Heatmap Korelasi Antar Fitur')
plt.show()

# 6. Preparation Data
X = df.drop(columns=['target'])
y = df['target']

# Normalisasi data fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Membagi data menjadi data latih dan data uji (70% latih, 30% uji)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
print("\n--- Ukuran Data Latih dan Uji ---")
print("Ukuran X_train:", X_train.shape)
print("Ukuran X_test:", X_test.shape)

# 7. Modeling & Evaluation
# Membuat dan melatih model SVM dengan kernel RBF
clf = SVC(kernel='rbf')
clf.fit(X_train, y_train)

# Melakukan prediksi pada data uji
y_pred = clf.predict(X_test)

# Mengevaluasi model
clf_acc = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\n--- Hasil Evaluasi Model SVM ---")
print("AKURASI SVM: {:.2f}%".format(clf_acc * 100))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(conf_matrix)
print(f"  True Negative (Sehat diprediksi Sehat): {conf_matrix[0][0]}")
print(f"  False Positive (Sehat diprediksi Sakit): {conf_matrix[0][1]}")
print(f"  False Negative (Sakit diprediksi Sehat): {conf_matrix[1][0]}")
print(f"  True Positive (Sakit diprediksi Sakit): {conf_matrix[1][1]}")

# Visualisasi Confusion Matrix dengan Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Sehat (0)', 'Sakit (1)'], yticklabels=['Sehat (0)', 'Sakit (1)'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# 8. Testing dengan Data Baru (Contoh)
print("\n--- Contoh Prediksi Data Baru ---")
new_data_dict = {
    'age': [50], 'sex': [1], 'cp': [2], 'trestbps': [130], 'chol': [250],
    'fbs': [0], 'restecg': [1], 'thalach': [170], 'exang': [0],
    'oldpeak': [1.5], 'slope': [2], 'ca': [0], 'thal': [2]
}

new_data = pd.DataFrame(new_data_dict)
print("Data Baru yang Akan Diprediksi:")
print(new_data)

# Normalisasi data baru menggunakan scaler yang sudah di-fit
scaled_new_data = scaler.transform(new_data)

# Prediksi data baru
y_pred_new = clf.predict(scaled_new_data)
hasil = "Terdeteksi Penyakit Jantung" if y_pred_new[0] == 1 else "Tidak Terdeteksi Penyakit Jantung"

print(f"\nHasil Diagnosis Data Baru: {hasil} (Label: {y_pred_new[0]})")
