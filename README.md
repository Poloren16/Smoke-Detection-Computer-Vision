# Sistem Deteksi Asap (Computer Vision)

Proyek ini adalah sistem Computer Vision untuk mendeteksi asap (dan api) menggunakan model YOLOv8 dari dataset yang telah disiapkan di folder `data`.

## Struktur Direktori
```
Computer Vision/
├── data/           # Berisi gambar dan label (test, train, val)
├── data.yaml       # Konfigurasi lokasi data untuk training
├── train.py        # Script untuk melatih model
├── app.py          # Aplikasi Antarmuka Pengguna dengan Streamlit
└── requirements.txt# Library Python yang dibutuhkan
```

## Prasyarat
Instal semua modul yang dibutuhkan dengan menjalankan perintah berikut di terminal:
```bash
pip install -r requirements.txt
```

## Cara Penggunaan

### 1. Training Model (Melatih Model Anda)
Sebelum menggunakan aplikasi GUI, Anda harus melatih model dengan dataset yang Anda miliki. Jalankan:
```bash
python train.py
```
Model ini menggunakan ultralytics YOLOv8. Proses ini mungkin memakan waktu tergantung pada banyaknya gambar dan spesifikasi perangkat Anda (sangat disarankan menggunakan GPU).
Setelah selesai, model terbaik akan tersimpan di lokasi:
`runs/detect/smoke_detection_model/weights/best.pt`

**Catatan:** Dalam script `data.yaml`, diasumsikan class `0` adalah asap (smoke) dan class `1` adalah api (fire). Jika urutannya berbeda pada dataset Anda, silakan ubah pada file tersebut.

### 2. Menjalankan Aplikasi Web
Setelah training selesai dan file `best.pt` berhasil dibuat, Anda dapat mengeksekusi aplikasi interface Streamlit:
```bash
streamlit run app.py
```
Akan terbuka halaman di browser Anda (biasanya di `http://localhost:8501`). Di sana, Anda bisa mengunggah file gambar dan sistem akan mendeteksi apakah terdapat asap dalam gambar tersebut.
