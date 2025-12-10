# Chassis Material Rating Predictor

Proyek ini adalah aplikasi Streamlit untuk memprediksi rating material (skala 1) untuk sasis otomotif berdasarkan properti mekanik (Su, Sy, E, G, bc, c1) dan tipe material. Ide dan pendekatan diadaptasi dari penelitian Nawale et al., *Design automation and CAD customization of an EV chassis* (J. Phys.: Conf. Ser. 2601, 012014, 2023).

## Fitur
- Antarmuka web sederhana dengan dua mode:
  - Mode 1: pilih tipe material, auto-fill properti elastis tipikal, lalu prediksi rating.
  - Mode 2: masukkan manual semua properti material, termasuk opsi satuan MPa/GPa untuk E dan G.
- Visualisasi distribusi keyakinan (probabilitas) untuk rating 1.
- Ringkasan input yang ditampilkan setelah prediksi.

## Persyaratan
Lihat `requirements.txt` (disediakan). Paket utama:
- streamlit
- pandas, numpy
- scikit-learn, joblib
- lightgbm
- plotly

## Menjalankan Aplikasi
1. (Opsional) Aktifkan virtual environment Anda.
2. Instal dependensi:
   ```powershell
   pip install -r requirements.txt
   ```
3. Jalankan aplikasi Streamlit:
   ```powershell
   streamlit run app3.py
   ```
4. Buka URL lokal yang ditampilkan oleh Streamlit (biasanya http://localhost:8501).

## Struktur Berkas Penting
- `app3.py`  aplikasi Streamlit utama.
- `requirements.txt`  daftar dependensi Python.
- `material_database.csv` dan `Data/Data Rating.csv`  data properti material dan label rating yang digunakan untuk pelatihan/prediksi.

## Referensi
- Nawale, P., Kanade, A., Nannaware, B., Sagalgile, A., Chougule, N., & Patange, A. (2023). Design automation and CAD customization of an EV chassis. *Journal of Physics: Conference Series*, 2601, 012014. https://doi.org/10.1088/1742-6596/2601/1/012014
