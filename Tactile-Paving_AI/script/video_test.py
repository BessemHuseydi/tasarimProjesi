from ultralytics import YOLO
import cv2
import os
from pathlib import Path

# ============================================
# AYARLAR
# ============================================

# Model yolu
MODEL_PATH = "tenji_training/yolov8m_800images/weights/best.pt"

# Video dosyası
VIDEO_PATH = "videos/v3.mp4"  # Kendi video yolunuzu yazın

# Çıktı ayarları
OUTPUT_DIR = "video_predictions"
OUTPUT_VIDEO = "output_video.mp4"

# Tespit ayarları
CONFIDENCE_THRESHOLD = 0.25    # Güven eşiği (0.1 - 0.9 arası)
IOU_THRESHOLD = 0.45           # NMS IOU eşiği
MAX_DETECTIONS = 300           # Maksimum tespit sayısı

# Görselleştirme ayarları
SHOW_LABELS = True             # Etiketleri göster
SHOW_CONF = True               # Güven skorlarını göster
LINE_WIDTH = 2                 # Kutu çizgi kalınlığı

# ============================================
# ANA FONKSİYONLAR
# ============================================

def test_video():
    """Video üzerinde YOLOv8 model testi"""
    
    # Model kontrolü
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model bulunamadı: {MODEL_PATH}")
        print("💡 MODEL_PATH değişkenini kontrol edin!")
        return
    
    # Video kontrolü
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Video bulunamadı: {VIDEO_PATH}")
        print("💡 VIDEO_PATH değişkenini kontrol edin!")
        return
    
    # Çıktı klasörü oluştur
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_VIDEO)
    
    print("=" * 60)
    print("🎬 YOLOv8 Video Testi Başlıyor")
    print("=" * 60)
    print(f"📦 Model: {MODEL_PATH}")
    print(f"🎥 Video: {VIDEO_PATH}")
    print(f"💾 Çıktı: {output_path}")
    print(f"🎯 Güven Eşiği: {CONFIDENCE_THRESHOLD}")
    print("=" * 60 + "\n")
    
    # Modeli yükle
    print("📥 Model yükleniyor...")
    model = YOLO(MODEL_PATH)
    print("✅ Model yüklendi!\n")
    
    # Video bilgilerini al
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Video Bilgileri:")
    print(f"   - FPS: {fps}")
    print(f"   - Çözünürlük: {width}x{height}")
    print(f"   - Toplam Frame: {total_frames}")
    print(f"   - Süre: {total_frames/fps:.2f} saniye\n")
    
    # Video writer oluştur
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    detection_count = 0
    
    print("🔄 Video işleniyor...")
    print("-" * 60)
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # YOLOv8 ile tespit
            results = model(
                frame,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                max_det=MAX_DETECTIONS,
                verbose=False
            )
            
            # Sonuçları görselleştir
            annotated_frame = results[0].plot(
                conf=SHOW_CONF,
                labels=SHOW_LABELS,
                line_width=LINE_WIDTH
            )
            
            # Tespit sayısını say
            detections = len(results[0].boxes)
            detection_count += detections
            
            # Frame'e bilgi ekle
            info_text = f"Frame: {frame_count}/{total_frames} | Tespit: {detections}"
            cv2.putText(
                annotated_frame, 
                info_text, 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2
            )
            
            # Çıktı videosuna yaz
            out.write(annotated_frame)
            
            # İlerleme göster
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"⏳ İlerleme: {progress:.1f}% ({frame_count}/{total_frames} frame)")
        
        print("-" * 60)
        print("✅ Video işleme tamamlandı!\n")
        
    except KeyboardInterrupt:
        print("\n⚠️ İşlem kullanıcı tarafından durduruldu!")
    
    finally:
        # Kaynakları serbest bırak
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    # İstatistikler
    print("=" * 60)
    print("📈 İSTATİSTİKLER")
    print("=" * 60)
    print(f"✅ İşlenen Frame Sayısı: {frame_count}")
    print(f"🎯 Toplam Tespit: {detection_count}")
    print(f"📊 Ortalama Tespit/Frame: {detection_count/frame_count:.2f}")
    print(f"💾 Çıktı Videosu: {output_path}")
    print("=" * 60)
    
    # Tespit detayları
    print("\n📋 TESPİT EDİLEN SINIFLAR:")
    print("-" * 60)
    
    # Sınıf isimlerini ve sayılarını topla
    class_counts = {}
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    cap.release()
    
    # Sınıfları yazdır
    if class_counts:
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   {class_name}: {count} tespit")
    else:
        print("   ❌ Hiç tespit yapılamadı!")
    
    print("-" * 60)
    print("\n🎉 Test tamamlandı!")
    print(f"💡 Çıktı videosunu izlemek için: {output_path}")

def save_sample_frames():
    """Videoden örnek frame'ler kaydet (GUI problemi için alternatif)"""
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model bulunamadı: {MODEL_PATH}")
        return
    
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Video bulunamadı: {VIDEO_PATH}")
        return
    
    # Çıktı klasörü
    frames_dir = os.path.join(OUTPUT_DIR, "sample_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    print("=" * 60)
    print("🖼️  YOLOv8 Örnek Frame Kaydetme")
    print("=" * 60)
    print(f"📦 Model: {MODEL_PATH}")
    print(f"🎥 Video: {VIDEO_PATH}")
    print(f"💾 Çıktı: {frames_dir}")
    print("=" * 60 + "\n")
    
    # Modeli yükle
    model = YOLO(MODEL_PATH)
    
    # Videoyu aç
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Her 30 frame'de bir kaydet (yaklaşık her saniye)
    frame_interval = 30
    saved_count = 0
    
    print("🔄 Örnek frame'ler kaydediliyor...\n")
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Belirli aralıklarla kaydet
        if frame_count % frame_interval == 0 or frame_count == 1:
            # Tespit yap
            results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
            annotated_frame = results[0].plot()
            
            # Kaydet
            output_path = os.path.join(frames_dir, f"frame_{frame_count:05d}.jpg")
            cv2.imwrite(output_path, annotated_frame)
            
            detections = len(results[0].boxes)
            print(f"✅ Frame {frame_count}/{total_frames} kaydedildi - {detections} tespit")
            saved_count += 1
    
    cap.release()
    
    print("\n" + "=" * 60)
    print(f"✅ {saved_count} örnek frame kaydedildi!")
    print(f"📁 Klasör: {frames_dir}")
    print("=" * 60)

# ============================================
# ÇALIŞTIR
# ============================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("YOLOv8 Video Test Menüsü")
    print("=" * 60)
    print("1. Video işle ve kaydet (TAM VIDEO)")
    print("2. Örnek frame'ler kaydet (HIZLI ÖNIZLEME)")
    print("=" * 60)
    
    choice = input("\nSeçiminiz (1/2): ").strip()
    
    if choice == "1":
        test_video()
    elif choice == "2":
        save_sample_frames()
    else:
        print("❌ Geçersiz seçim! (1 veya 2)")
        print("💡 Direkt video işlemek için: test_video()")
        test_video()  # Varsayılan olarak video işle