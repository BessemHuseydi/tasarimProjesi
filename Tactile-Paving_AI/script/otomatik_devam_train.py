from ultralytics import YOLO
import torch
import os
import glob

# GPU kontrolü
print("=" * 50)
print("Sistem Bilgileri")
print("=" * 50)
print(f"CUDA Kullanılabilir mi: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Bellek: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print("=" * 50)

# Model yolu
MODEL_NAME = "yolov8m.pt"

# Veri seti yaml dosyası
DATA_YAML = "merged_tenji_dataset/data.yaml"

# Eğitim parametreleri
EPOCHS = 100              # Epoch sayısı (800 resim için 100-150 önerilir)
BATCH_SIZE = 16           # Batch size (GPU belleğinize göre ayarlayın: 8, 16, 32)
IMAGE_SIZE = 640          # Görsel boyutu
PATIENCE = 20             # Early stopping patience
WORKERS = 8               # Veri yükleme thread sayısı

# Optimizer ve Learning Rate ayarları
OPTIMIZER = 'AdamW'       # 'SGD', 'Adam', 'AdamW', 'RMSProp'
LEARNING_RATE = 0.001     # İlk learning rate (küçük veri seti için biraz daha yüksek)
LRF = 0.01                # Final learning rate oranı (lr * lrf)

# Augmentation parametreleri
DEGREES = 10.0            # Görsel rotasyon (±degrees)
TRANSLATE = 0.1           # Görsel kaydırma (0.1 = %10)
SCALE = 0.5               # Görsel ölçekleme (0.5 = %50)
SHEAR = 0.0               # Görsel eğme
PERSPECTIVE = 0.0         # Perspektif değişimi
FLIPUD = 0.0              # Dikey flip olasılığı
FLIPLR = 0.5              # Yatay flip olasılığı (0.5 = %50)
MOSAIC = 1.0              # Mosaic augmentation
MIXUP = 0.0               # Mixup augmentation
HSV_H = 0.015             # Hue augmentation
HSV_S = 0.7               # Saturation augmentation
HSV_V = 0.4               # Value augmentation

# Çıktı klasörü
PROJECT_NAME = "tenji_training"
RUN_NAME = "yolov8m_800images"

def find_last_checkpoint():
    """Son checkpoint'i bul"""
    checkpoint_path = f"{PROJECT_NAME}/{RUN_NAME}/weights/last.pt"
    
    if os.path.exists(checkpoint_path):
        print(f"\n✅ Checkpoint bulundu: {checkpoint_path}")
        return checkpoint_path
    
    # Alternatif: Eğer run_name farklıysa, tüm last.pt dosyalarını ara
    pattern = f"{PROJECT_NAME}/*/weights/last.pt"
    checkpoints = glob.glob(pattern)
    
    if checkpoints:
        # En son değiştirilmiş checkpoint'i al
        latest_checkpoint = max(checkpoints, key=os.path.getmtime)
        print(f"\n✅ Checkpoint bulundu: {latest_checkpoint}")
        return latest_checkpoint
    
    print("\n❌ Checkpoint bulunamadı. Yeni eğitim başlatılacak.")
    return None

def train_yolov8(resume=True):
    """YOLOv8m modelini eğit veya devam ettir"""
    
    # Checkpoint kontrolü
    checkpoint = None
    if resume:
        checkpoint = find_last_checkpoint()
    
    print("\n" + "=" * 50)
    if checkpoint:
        print("YOLOv8m Eğitim DEVAM EDİYOR")
        print("=" * 50)
        print(f"📂 Checkpoint: {checkpoint}")
    else:
        print("YOLOv8m YENİ Eğitim Başlıyor")
        print("=" * 50)
        print(f"Model: {MODEL_NAME}")
    
    print(f"Veri Seti: {DATA_YAML}")
    print(f"Toplam Epoch: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Görsel Boyutu: {IMAGE_SIZE}")
    print(f"Optimizer: {OPTIMIZER}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print("=" * 50 + "\n")
    
    try:
        # Checkpoint'ten devam et veya yeni model yükle
        if checkpoint:
            model = YOLO(checkpoint)
            print("🔄 Eğitim kaldığı yerden devam ediyor...")
        else:
            model = YOLO(MODEL_NAME)
            print("🆕 Yeni eğitim başlatılıyor...")
        
        # Eğitimi başlat/devam ettir
        results = model.train(
            # Checkpoint'ten devam için önemli!
            resume=checkpoint is not None,
            
            # Veri ve model
            data=DATA_YAML,
            epochs=EPOCHS,
            batch=BATCH_SIZE,
            imgsz=IMAGE_SIZE,
            
            # Çıktı ayarları
            project=PROJECT_NAME,
            name=RUN_NAME,
            exist_ok=True,  # Checkpoint için True olmalı
            
            # Optimizer ayarları
            optimizer=OPTIMIZER,
            lr0=LEARNING_RATE,
            lrf=LRF,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            
            # Training ayarları
            patience=PATIENCE,
            save=True,
            save_period=10,        # Her 10 epoch'ta bir kaydet
            cache=False,           # RAM'de cache (True yapabilirsiniz)
            device='cpu' if not torch.cuda.is_available() else 0,
            workers=WORKERS,
            pretrained=True if not checkpoint else False,  # Checkpoint varsa pretrained False
            verbose=True,
            seed=0,
            deterministic=True,
            single_cls=False,
            rect=False,
            cos_lr=False,
            close_mosaic=10,
            
            # Augmentation ayarları
            degrees=DEGREES,
            translate=TRANSLATE,
            scale=SCALE,
            shear=SHEAR,
            perspective=PERSPECTIVE,
            flipud=FLIPUD,
            fliplr=FLIPLR,
            mosaic=MOSAIC,
            mixup=MIXUP,
            hsv_h=HSV_H,
            hsv_s=HSV_S,
            hsv_v=HSV_V,
            
            # Loss fonksiyonu ağırlıkları
            box=7.5,
            cls=0.5,
            dfl=1.5,
            
            # Validation ayarları
            val=True,
            plots=True,
            
            # Multi-scale training
            fraction=1.0,
        )
        
        print("\n" + "=" * 50)
        print("✅ Eğitim Tamamlandı!")
        print("=" * 50)
        print(f"📁 Sonuçlar: {PROJECT_NAME}/{RUN_NAME}")
        print(f"🎯 En iyi model: {PROJECT_NAME}/{RUN_NAME}/weights/best.pt")
        print(f"📊 Metrikler: {PROJECT_NAME}/{RUN_NAME}/results.png")
        print("=" * 50)
        
        return results
        
    except KeyboardInterrupt:
        print("\n⚠️ Eğitim kullanıcı tarafından durduruldu!")
        print(f"💾 Checkpoint kaydedildi: {PROJECT_NAME}/{RUN_NAME}/weights/last.pt")
        print(f"🔄 Devam etmek için: python {__file__}")
        return None
        
    except Exception as e:
        print(f"\n❌ Hata oluştu: {e}")
        print(f"💾 Checkpoint kaydedildi: {PROJECT_NAME}/{RUN_NAME}/weights/last.pt")
        print(f"🔄 Devam etmek için: python {__file__}")
        raise

def validate_model(model_path):
    """Modeli test seti üzerinde değerlendir"""
    print("\n" + "=" * 50)
    print("Model Değerlendirme")
    print("=" * 50)
    
    try:
        model = YOLO(model_path)
        metrics = model.val(data=DATA_YAML, split='test')
        
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall: {metrics.box.mr:.4f}")
        print("=" * 50)
        
        return metrics
    except Exception as e:
        print(f"❌ Değerlendirme hatası: {e}")
        return None

if __name__ == "__main__":
    # Eğitimi başlat veya devam ettir (resume=True otomatik devam eder)
    results = train_yolov8(resume=True)
    
    # Eğitim tamamlandıysa, en iyi modeli test et
    if results is not None:
        best_model_path = f"{PROJECT_NAME}/{RUN_NAME}/weights/best.pt"
        if os.path.exists(best_model_path):
            validate_model(best_model_path)
        
        print("\n🎉 İşlem tamamlandı!")
        print(f"💡 Tahmin için: python predict.py --weights {best_model_path}")
    else:
        print("\n⏸️ Eğitim tamamlanmadı.")
        print(f"🔄 Devam etmek için kodu tekrar çalıştırın: python {__file__}")