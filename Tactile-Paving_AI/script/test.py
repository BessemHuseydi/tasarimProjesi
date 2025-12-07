# import cv2
# import time
# from ultralytics import YOLO

# model = YOLO(r"C:\Users\besse\coding\tasarim\runs\detect\yolov8L_Tactile_model_400_Data13\weights\best.pt")
# video_path = r"C:\Users\besse\coding\tasarim\Tactile-Paving_AI\videos\v5.mp4"
# cap = cv2.VideoCapture(video_path)

# # Elle FPS değeri ver
# fps_output = 30.0
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# output_path = r"C:\Users\besse\coding\tasarim\Tactile-Paving_AI\videos\output_v4_detected.mp4"
# out = cv2.VideoWriter(output_path, fourcc, fps_output, (width, height))

# prev_time = time.time()

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Model ile tahmin
#     results = model(frame, verbose=False)
#     annotated_frame = results[0].plot()

#     # FPS hesapla
#     curr_time = time.time()
#     fps = 1 / (curr_time - prev_time)
#     prev_time = curr_time

#     # FPS yazısı
#     fps_text = f"FPS: {fps:.2f}"
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 0.6
#     font_thickness = 2
#     (text_width, text_height), _ = cv2.getTextSize(fps_text, font, font_scale, font_thickness)
#     cv2.rectangle(annotated_frame, (10, 10), (10 + text_width + 10, 10 + text_height + 10), (0, 255, 0), -1)
#     cv2.putText(annotated_frame, fps_text, (15, 10 + text_height), font, font_scale, (0, 0, 0), font_thickness)

#     # Görüntüyü yaz ve göster
#     out.write(annotated_frame)
#     cv2.imshow("YOLOv8 Detection", annotated_frame)

#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break

# cap.release()
# out.release()
# cv2.destroyAllWindows()















import cv2
import time
from ultralytics import YOLO

# YOLOv8l modelini yükle (path'i senin model dosyan)
model = YOLO(r"C:\Users\besse\coding\tasarim\Tactile-Paving_AI\models\tactile_v8L_400.pt")  # <-- Buraya kendi model .pt dosyanı yaz

# Video dosyasını aç (path'i senin videon)
video_path = r"C:\Users\besse\coding\tasarim\Tactile-Paving_AI\videos\v6.mp4"  # <-- Buraya kendi video dosyanı yaz
cap = cv2.VideoCapture(video_path)

# FPS hesaplama için zaman
prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # FPS hesaplama
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    # Model ile tahmin
    results = model(frame, verbose=False)

    # Sonuçları çiz
    annotated_frame = results[0].plot()

    # FPS metni
    fps_text = f"FPS: {fps:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2

    # Yazı boyutunu al
    (text_width, text_height), _ = cv2.getTextSize(fps_text, font, font_scale, font_thickness)

    # Arka plan dikdörtgeni (yeşil)
    cv2.rectangle(annotated_frame, (10, 10), (10 + text_width + 10, 10 + text_height + 10), (0, 255, 0), -1)

    # Yazıyı çiz (siyah)
    cv2.putText(annotated_frame, fps_text, (15, 10 + text_height), font, font_scale, (0, 0, 0), font_thickness)

    # Göster
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
