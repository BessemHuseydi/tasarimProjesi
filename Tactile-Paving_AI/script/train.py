# from ultralytics import YOLO
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# # Ana program bloğu
# if __name__ == '__main__':
#     # YOLOv8 tespit modeli yükle
#     model = YOLO("yolov8l.pt")

#     # Eğitim verileri YAML yolu
#     data_path = r"C:\Users\besse\coding\tasarim\Tactile-Paving_AI\tactile_data\data.yaml"

#     # Eğitim parametreleri
#     remaining_epochs = 150

#     # Modeli eğit
#     model.train(
#         data=data_path,
#         task="detect",               
#         epochs=remaining_epochs,
#         patience=20,
#         imgsz=640,
#         workers=8,
#         batch=8,
#         device=0,
#         name="yolov8L_Tactile_model_400_Data",
#         resume=False
#     )

#     print("Eğitim tamamlandı!")
from ultralytics import YOLO
import os

# OpenMP hatası çözümü (bazı sistemlerde gerekli olabilir)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    # ✅ 1. Segmentasyon modeli yolunu belirt (önceden eğitilmiş YOLOv8-seg modeli)
    model_path = "yolov8l-seg.pt"  # veya yolov8m-seg.pt, yolov8x-seg.pt seçebilirsin
    model = YOLO(model_path)

    # ✅ 2. data.yaml dosya yolu (mutlaka segmentasyon maskeleriyle uyumlu olmalı)
    data_yaml_path = r"C:\Users\besse\coding\tasarim\Tactile-Paving_AI\data\tactile paving image dataset.v3i.yolov8\data.yaml"


    # ✅ 3. Eğitim parametreleri
    model.train(
        data=data_yaml_path,
        task="segment",                     # Segmentasyon görevini belirtiyoruz
        epochs=50,
        patience=20,
        imgsz=640,
        workers=8,
        batch=8,
        device=0,                           # 0 = GPU, 'cpu' yazarsan CPU'da çalışır
        name="yolov8L_Tactile_Segment",    # Eğitim çıktısı klasör adı
        resume=False
    )

    print("✅ SEGMENTASYON eğitimi tamamlandı!")

if __name__ == "__main__":
    main()
