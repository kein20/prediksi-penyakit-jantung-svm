# Prediksi Penyakit Jantung Menggunakan Support Vector Machine (SVM)

Proyek ini adalah implementasi metode *machine learning* untuk memprediksi risiko penyakit jantung pada pasien. [cite_start]Model yang digunakan adalah **Support Vector Machine (SVM)** dengan kernel RBF, dibangun sebagai Proyek Akhir mata kuliah Kecerdasan Buatan di Universitas Negeri Surabaya. [cite: 5, 13]

---

### 📋 Deskripsi Proyek

[cite_start]Penyakit jantung adalah salah satu penyebab utama kematian di seluruh dunia. [cite: 22] [cite_start]Deteksi dini menjadi kunci untuk penanganan yang efektif dan mengurangi risiko komplikasi. [cite: 25, 26] Proyek ini bertujuan untuk membangun sebuah model klasifikasi yang akurat untuk membantu mendiagnosis risiko penyakit jantung berdasarkan data atribut medis pasien.

[cite_start]Model ini berhasil mencapai **akurasi sebesar 82.42%** pada data pengujian. [cite: 1059]

---

### 📊 Dataset

[cite_start]Dataset yang digunakan dalam proyek ini berasal dari **UCI Machine Learning Repository**, yaitu dataset "Heart Disease". [cite: 96] [cite_start]Dataset ini berisi 14 atribut medis utama dari 303 sampel data pasien yang unik setelah melalui proses pembersihan data duplikat. [cite: 100, 150]

**Atribut Utama yang Digunakan:**
* [cite_start]`age`: Usia pasien [cite: 106]
* [cite_start]`sex`: Jenis kelamin (0 = perempuan, 1 = laki-laki) [cite: 116]
* [cite_start]`cp`: Tipe nyeri dada (*chest pain type*) [cite: 110]
* [cite_start]`trestbps`: Tekanan darah saat istirahat (*resting blood pressure*) [cite: 112]
* [cite_start]`chol`: Kadar kolesterol (*serum cholestoral*) [cite: 113]
* [cite_start]`fbs`: Gula darah puasa > 120 mg/dl (*fasting blood sugar*) [cite: 114]
* [cite_start]`restecg`: Hasil elektrokardiografi [cite: 115]
* [cite_start]`thalach`: Detak jantung maksimum (*max heart rate achieved*) [cite: 123]
* [cite_start]`exang`: Angina akibat olahraga (*exercise induced angina*) [cite: 124]
* [cite_start]`oldpeak`: Depresi segmen ST [cite: 124]
* [cite_start]`slope`: Kemiringan segmen ST [cite: 126]
* [cite_start]`ca`: Jumlah pembuluh darah utama yang terlihat [cite: 128]
* [cite_start]`thal`: Hasil *thalium scan* [cite: 130]
* [cite_start]`target`: Variabel target (0 = tidak ada penyakit jantung, 1 = ada penyakit jantung) [cite: 103]

---

### ⚙️ Metodologi

Proses pembangunan model mengikuti langkah-langkah berikut:
1.  [cite_start]**Pembersihan Data:** Menghapus data duplikat dari dataset awal. [cite: 162]
2.  [cite_start]**Preprocessing Data:** Melakukan normalisasi data menggunakan `StandardScaler` agar setiap fitur memiliki skala yang seragam. [cite: 164]
3.  [cite_start]**Pembagian Data:** Dataset dibagi menjadi 70% data latih (*training data*) dan 30% data uji (*testing data*). [cite: 153, 154]
4.  [cite_start]**Pelatihan Model:** Model **Support Vector Machine (SVM)** dengan kernel **Radial Basis Function (RBF)** dilatih menggunakan data latih. [cite: 173]
5.  [cite_start]**Evaluasi Model:** Kinerja model dievaluasi menggunakan data uji dengan metrik seperti *Accuracy*, *Precision*, *Recall*, *F1-Score*, dan *Confusion Matrix*. [cite: 179]

---

### 🚀 Hasil

[cite_start]Model yang telah dilatih berhasil mencapai **akurasi 82.42%**[cite: 1059]. Berikut adalah detail dari *confusion matrix* pada data uji:
* [cite_start]**True Positive:** 39 [cite: 1065]
* [cite_start]**True Negative:** 36 [cite: 1063]
* [cite_start]**False Positive:** 12 [cite: 1063]
* [cite_start]**False Negative:** 4 [cite: 1064]

---

### 📂 File dalam Repository

* `svm_heart_disease.py`: File utama berisi kode Python untuk melatih dan mengevaluasi model SVM.
* `heart_disease_data.csv`: Dataset utama yang digunakan untuk analisis.
* `training_data.csv`: Subset data yang digunakan untuk melatih model.
* `testing_data.csv`: Subset data yang digunakan untuk menguji model.
