from ultralytics import YOLO

def main():
    # Inisialisasi model YOLOv8 (bisa diganti yolov8n.pt untuk model yang lebih ringan)
    model = YOLO('yolov8n.pt') 

    # Melakukan training model dengan dataset yang ada
    # Epoch diset ke 50 sebagai contoh, batch size menyesuaikan memori GPU
    results = model.train(
        data='data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        name='smoke_detection_model'
    )

    print("Training selesai. Model tersimpan di folder 'runs/detect/smoke_detection_model/weights/best.pt'")

if __name__ == '__main__':
    main()
