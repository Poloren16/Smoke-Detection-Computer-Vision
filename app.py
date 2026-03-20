import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2
import os

st.set_page_config(page_title="Smoke Detection CV", layout="wide")
st.title("Sistem Deteksi Asap dengan YOLOv8")
st.write("Aplikasi untuk mendeteksi asap pada gambar menggunakan model yang dilatih kustom.")

import glob

@st.cache_resource
def load_model():
    # Cek apakah model hasil training sudah ada
    latest_model = 'runs/detect/smoke_detection_model2/weights/best.pt'
    if os.path.exists(latest_model):
        model = YOLO(latest_model)
        # Meniban nama label yang tersimpan di dalam best.pt
        model.model.names = {0: 'Asap', 1: 'Api'}
        return model, True
    else:
        # Fallback ke model YOLOv8n bawaan (hanya deteksi objek umum, tanpa asap)
        return YOLO('yolov8n.pt'), False

model, is_custom = load_model()

if not is_custom:
    st.warning("⚠️ Model deteksi Asap ('best.pt') belum ditemukan karena belum di-training!")
    st.info("Saat ini aplikasi menggunakan model bawaan polos yang hanya bisa mendeteksi 80 objek umum (Manusia, Mobil, dsb.) dan **TIDAK BISA** mendeteksi asap.")
else:
    st.success("✅ Model deteksi Asap ('best.pt') berhasil dimuat!")

# Pilihan Mode Input
option = st.radio("Pilih Mode Input Gambar:", ("Unggah dari Komputer", "Ambil Foto dari Kamera Laptop"))

image = None
if option == "Unggah dari Komputer":
    uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
else:
    camera_photo = st.camera_input("Ambil Foto Langsung")
    if camera_photo is not None:
        image = Image.open(camera_photo).convert('RGB')

if image is not None:
    st.image(image, caption='Gambar Input', use_container_width=True)
    
    if st.button('Deteksi Asap & Api') and model is not None:
        with st.spinner('Memproses...'):
            # Merubah gambar menjadi format array agar bisa dibaca model
            img_array = np.array(image)
            
            # Melakukan prediksi
            results = model.predict(source=img_array, conf=0.60)
            
            # Menampilkan hasil deteksi
            for r in results:
                # Dapatkan gambar beranotasi
                annotated_img = r.plot()
                # Tampilkan ke layar tanpa convert warna karena input awalnya sudah RGB dari PIL
                st.image(annotated_img, caption='Hasil Deteksi', use_container_width=True)
                
                # Tampilkan tabel deteksi
                if len(r.boxes) > 0:
                    st.write(f"Ditemukan **{len(r.boxes)}** objek:")
                    for box in r.boxes:
                        class_id = int(box.cls[0].item())
                        class_name = model.names[class_id]
                        conf = round(box.conf[0].item(), 2)
                        st.write(f"- {class_name} (Confidence: {conf})")
                else:
                    st.write("Tidak mendeteksi asap atau api.")
