# Sign Language Recognition

Proyek ini menggunakan Python dan OpenCV untuk melakukan pengenalan bahasa isyarat (Sign Language Recognition) secara otomatis menggunakan model machine learning. Anda dapat menjalankan inference secara lokal untuk mendeteksi dan mengenali bahasa isyarat dari input video/camera.

## Fitur
- Deteksi bahasa isyarat menggunakan webcam atau file video
- Berbasis Python + OpenCV
- Mudah dijalankan secara lokal

## Persyaratan
- Python 3.7 atau lebih baru
- pip (Python package manager)

## Instalasi

1. **Clone repository ini**
   ```bash
   git clone https://github.com/7z1x/Sign_language.git
   cd Sign_language
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Cara Menjalankan

Jalankan file inference:
```bash
python inference_final.py
```
Biasanya aplikasi akan otomatis membuka webcam dan menampilkan hasil deteksi pada jendela baru. Pastikan webcam Anda aktif.

## Catatan
- Jika Anda ingin menggunakan file video, edit script `inference_final.py` untuk mengambil input dari file alih-alih webcam.
- Pastikan semua dependensi pada `requirements.txt` sudah terinstall.

## Kontribusi
Pull request sangat dipersilakan! Silakan open issue jika Anda menemukan bug atau ingin menambah fitur.

## Lisensi
Silakan tambahkan lisensi sesuai kebutuhan Anda (MIT, GPL, dsb).

---

Happy coding! 👋