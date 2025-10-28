# Prediksi Penyakit Jantung Menggunakan Support Vector Machine (SVM)

Proyek ini adalah implementasi metode *machine learning* untuk memprediksi risiko penyakit jantung pada pasien. Model yang digunakan adalah **Support Vector Machine (SVM)** dengan kernel RBF.
---

### 📋 Deskripsi Proyek

Penyakit jantung adalah salah satu penyebab utama kematian di seluruh dunia. Deteksi dini menjadi kunci untuk penanganan yang efektif dan mengurangi risiko komplikasi. Proyek ini bertujuan untuk membangun sebuah model klasifikasi yang akurat untuk membantu mendiagnosis risiko penyakit jantung berdasarkan data atribut medis pasien.

Model ini berhasil mencapai **akurasi sebesar 82.42%** pada data pengujian.

---

### 📊 Dataset

Dataset yang digunakan dalam proyek ini berasal dari **UCI Machine Learning Repository**, yaitu dataset "Heart Disease". Dataset ini berisi 14 atribut medis utama dari 303 sampel data pasien yang unik setelah melalui proses pembersihan data duplikat.

**Atribut Utama yang Digunakan:**
* `age`: Usia pasien
* `sex`: Jenis kelamin (0 = perempuan, 1 = laki-laki)
* `cp`: Tipe nyeri dada (*chest pain type*)
* `trestbps`: Tekanan darah saat istirahat (*resting blood pressure*)
* `chol`: Kadar kolesterol (*serum cholestoral*)
* `fbs`: Gula darah puasa > 120 mg/dl (*fasting blood sugar*)
* `restecg`: Hasil elektrokardiografi
* `thalach`: Detak jantung maksimum (*max heart rate achieved*)
* `exang`: Angina akibat olahraga (*exercise induced angina*)
* `oldpeak`: Depresi segmen ST
* `slope`: Kemiringan segmen ST
* `ca`: Jumlah pembuluh darah utama yang terlihat
* `thal`: Hasil *thalium scan*
* `target`: Variabel target (0 = tidak ada penyakit jantung, 1 = ada penyakit jantung)

---

### ⚙️ Metodologi

Proses pembangunan model mengikuti langkah-langkah berikut:
1.  **Pembersihan Data:** Menghapus data duplikat dari dataset awal.
2.  **Preprocessing Data:** Melakukan normalisasi data menggunakan `StandardScaler` agar setiap fitur memiliki skala yang seragam.
3.  **Pembagian Data:** Dataset dibagi menjadi 70% data latih (*training data*) dan 30% data uji (*testing data*).
4.  **Pelatihan Model:** Model **Support Vector Machine (SVM)** dengan kernel **Radial Basis Function (RBF)** dilatih menggunakan data latih.
5.  **Evaluasi Model:** Kinerja model dievaluasi menggunakan data uji dengan metrik seperti *Accuracy*, *Precision*, *Recall*, *F1-Score*, dan *Confusion Matrix*.

---

### 🚀 Hasil

Model yang telah dilatih berhasil mencapai **akurasi 82.42%**. Berikut adalah detail dari *confusion matrix* pada data uji:
* **True Positive:** 39
* **True Negative:** 36
* **False Positive:** 12
* **False Negative:** 4

---

### 📂 File dalam Repository

* `svm_heart_disease.py`: File utama berisi kode Python untuk melatih dan mengevaluasi model SVM.
* `heart_disease_data.csv`: Dataset utama yang digunakan untuk analisis.
* `training_data.csv`: Subset data yang digunakan untuk melatih model.
* `testing_data.csv`: Subset data yang digunakan untuk menguji model.
