import os
import cv2

# Direktori untuk menyimpan data
DATA_DIR = './dataset_1_tangan'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Jumlah kelas/isyarat yang akan dikumpulkan
number_of_classes = 10
# Jumlah gambar per kelas
dataset_size = 500

# Buka kamera (indeks 0 biasanya webcam internal)
cap = cv2.VideoCapture(2)

# Loop untuk setiap kelas
for j in range(number_of_classes):
    # Buat direktori untuk kelas jika belum ada
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Mengumpulkan data untuk kelas: {}'.format(j))

    # Tampilkan pesan "Ready?" sampai pengguna menekan 'q'
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari kamera.")
            break
        
        cv2.putText(frame, 'Siap? Tekan "Q" untuk memulai!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Mulai mengambil gambar
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Tampilkan frame saat pengambilan gambar
        cv2.imshow('frame', frame)
        cv2.waitKey(25) # Beri jeda singkat
        
        # Simpan gambar
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)

        counter += 1
        print(f'Gambar ke-{counter} untuk kelas {j} disimpan')

        # ================= BLOK JEDA YANG DIPERJELAS =================
        # Cek apakah sudah mencapai kelipatan 100 dan belum selesai sepenuhnya
        if counter % 100 == 0 and counter < dataset_size:
            print(f"Batch ke-{counter // 100} dari {dataset_size // 100} selesai. Jeda...")
            
            # Tampilkan pesan jeda yang informatif dan tunggu tombol 'q' untuk melanjutkan
            while True:
                # Baca frame terus menerus agar video tidak freeze dan Anda bisa atur pose
                ret, pause_frame = cap.read()
                if not ret:
                    break
                
                # Definisikan teks pesan
                line1 = f"Batch {counter // 100} Selesai. Atur pose Anda."
                line2 = "Tekan 'Q' untuk Lanjut..."

                # Tampilkan teks di layar
                cv2.putText(pause_frame, line1, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(pause_frame, line2, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                cv2.imshow('frame', pause_frame)
                
                # Tunggu tombol 'q' untuk keluar dari mode jeda
                if cv2.waitKey(25) == ord('q'):
                    break
        # =============================================================

print('Pengumpulan data selesai.')
cap.release()
cv2.destroyAllWindows()