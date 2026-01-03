# import cv2
# import time
# import pyttsx3
# from ultralytics import YOLO

# # TTS motoru
# engine = pyttsx3.init(driverName='sapi5')
# engine.setProperty('rate', 150)
# engine.setProperty('volume', 1.0)

# def speak(text):
#     engine.say(text)
#     engine.runAndWait()

# # YOLO modeli
# model = YOLO(r"C:\Users\besse\coding\tasarim\runs\detect\yolov8L_Tactile_model_400_Data13\weights\best.pt")

# # Video
# video_path = r"C:\Users\besse\coding\tasarim\Tactile-Paving_AI\videos\v6.mp4"
# cap = cv2.VideoCapture(video_path)

# prev_time = 0
# last_spoken_time = 0
# speak_interval = 3  # saniye

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     curr_time = time.time()
#     fps = 1 / (curr_time - prev_time) if prev_time else 0
#     prev_time = curr_time

#     results = model(frame, verbose=False)
#     annotated_frame = results[0].plot()

#     # Etiket kontrol
#     detected = False
#     for box in results[0].boxes:
#         class_id = int(box.cls[0])
#         class_name = model.names[class_id]
#         print(f"Algılanan: {class_name}")  # kontrol için log
#         if "kılavuz" in class_name.lower():
#             detected = True
#             break

#     # Sesli uyarı
#     if detected and (time.time() - last_spoken_time > speak_interval):
#         speak("Sarı kılavuz algılandı.")
#         last_spoken_time = time.time()

#     # Görüntüyü göster
#     cv2.imshow("YOLOv8 Detection with Sound", annotated_frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
from ultralytics import YOLO
import cv2

# Eğitilmiş segmentasyon modelini yükle
model = YOLO(r"C:\Users\besse\coding\tasarim\Tactile-Paving_AI\models\yolov8L_Tactile_Segment6\weights\best.pt")

# Test edilecek video yolu (KENDİ DOSYA YOLUNU YAZ)
video_path = r"C:\Users\besse\coding\tasarim\Tactile-Paving_AI\videos\v6.mp4"

# Tahmin yap (segmentasyon ile)
model.predict(
    source=video_path,  # video dosya yolu
    show=True,          # ekranda anlık göster
    save=True,          # çıktıyı kaydet
    conf=0.4,           # minimum güven skoru (isteğe bağlı)
    iou=0.5,            # segmentasyon için IOU eşiği
    name="test_output", # çıktılar "runs/segment/test_output" klasörüne kaydedilir
    stream=False        # video için stream kapalı
)
