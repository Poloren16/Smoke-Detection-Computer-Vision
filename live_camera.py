import cv2
import os
from ultralytics import YOLO

import glob

def main():
    print("Membuka kamera...")
    
    # 1. Cek model
    latest_model = 'runs/detect/smoke_detection_model2/weights/best.pt'
    if os.path.exists(latest_model):
        print(f"✅ Memuat model Asap dari ({latest_model})...")
        model = YOLO(latest_model)
        # Meniban nama label yang tersimpan di dalam best.pt
        model.model.names = {0: 'Asap', 1: 'Api'}
    else:
        print("⚠️ Model 'best.pt' belum ada! Fallback menggunakan model bawaan polos (yolov8n.pt).")
        print("⚠️ Model bawaan HANYA bisa mendeteksi 80 objek COCO (orang, mobil, kucing, dll).")
        model = YOLO('yolov8n.pt')

    # 2. Buka kamera (index 0 biasanya adalah webcam utama laptop/PC)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Gagal membuka kamera. Pastikan webcam terhubung dan tidak sedang digunakan aplikasi lain.")
        return

    print("Kamera berhasil dibuka! Tekan tombol 'q' pada keyboard untuk keluar/menutup jendela.")

    # [BARU] Setup VideoWriter untuk merekam hasil kamera
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps == 0:
        fps = 20.0  # fallback jika FPS webcam tidak terbaca
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec untuk format MP4
    out_video = cv2.VideoWriter('hasil_rekaman_cv.mp4', fourcc, fps, (frame_width, frame_height))
    print("🔴 Perekaman video DIMULAI. Video nanti akan disimpan dengan nama 'hasil_rekaman_cv.mp4'.")

    # 3. Looping untuk membaca video secara real-time
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal mengambil frame/gambar dari kamera.")
            break

        # 4. Deteksi objek pada frame (gambar) saat ini
        # conf=0.5 artinya kita hanya percaya deteksi di atas 50% yakin
        results = model.predict(source=frame, conf=0.60, verbose=False)

        # 5. Gambarkan kotak hasil prediksi ke atas frame
        # r.plot() akan mengembalikan array gabungan gambar + kotak deteksi
        annotated_frame = results[0].plot()

        # [BARU] Simpan setiap detik gambar (frame) ke dalam file video MP4
        out_video.write(annotated_frame)

        # 6. Tampilkan ke layar
        cv2.imshow("Sistem Deteksi Kamera Langsung (Real-Time)", annotated_frame)

        # Keluar dari loop jika user menekan tombol 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 7. Bersihkan resource setelah selesai
    cap.release()
    out_video.release() # [BARU] Menyimpan & menutup rekaman video permanen
    cv2.destroyAllWindows()
    print("⏹️ Rekaman berhasil dihentikan. Silakan buka file 'hasil_rekaman_cv.mp4' di folder ini!")

if __name__ == "__main__":
    main()
