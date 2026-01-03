import cv2
import pytesseract
import pyttsx3
import threading
import queue
from collections import deque
import time
import os
import sys

# Windows kullanıyorsanız ve Tesseract path hatası alırsanız alttaki satırı aktif edip kendi yolunuzu yazın:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class UniversalOCRReader:
    def __init__(self):
        self.tesseract_lang = 'tur+eng'
        self.frame_skip = 10  # Her 10 frame'de bir işle
        self.detected_texts = deque(maxlen=10)
        self.frame_count = 0
        
        # Ses motoru
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 1.0)
        
        # Türkçe ses ayarı
        voices = self.engine.getProperty('voices')
        for voice in voices:
            if 'turkish' in voice.name.lower() or 'tr' in voice.id.lower():
                self.engine.setProperty('voice', voice.id)
                break
        
        # Thread-safe kuyruk
        self.text_queue = queue.Queue()
        self.speaking = False
        
        print("\n" + "="*60)
        print("💻 SİSTEM: SADECE CPU MODU AKTİF")
        print("="*60 + "\n")
    
    def preprocess_frame(self, frame):
        """CPU ile frame ön işleme"""
        # Gri tonlamaya çevir
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Gürültü azaltma (Blur)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Eşikleme (Threshold)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Morfolojik işlemler (Gürültü temizleme)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return morphed
    
    def detect_text_with_boxes(self, frame):
        """Frame'den metin ve konumlarını çıkarma"""
        processed = self.preprocess_frame(frame)
        
        # OCR uygula
        custom_config = r'--oem 3 --psm 6'
        ocr_data = pytesseract.image_to_data(
            processed, 
            lang=self.tesseract_lang, 
            config=custom_config,
            output_type=pytesseract.Output.DICT
        )
        
        return ocr_data
    
    def draw_boxes_and_text(self, frame, ocr_data):
        """Frame üzerine tespit edilen metinleri çiz"""
        n_boxes = len(ocr_data['text'])
        detected_text = []
        
        for i in range(n_boxes):
            conf = int(ocr_data['conf'][i])
            # Güven oranı %30'dan büyükse işle
            if conf > 30:
                text = ocr_data['text'][i].strip()
                if len(text) > 1:
                    detected_text.append(text)
                    
                    x = ocr_data['left'][i]
                    y = ocr_data['top'][i]
                    w = ocr_data['width'][i]
                    h = ocr_data['height'][i]
                    
                    # Yeşil dikdörtgen
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Metin yazısı
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (x, y - text_size[1] - 8), 
                                (x + text_size[0] + 4, y), (0, 255, 0), -1)
                    cv2.putText(frame, text, (x + 2, y - 4),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        full_text = ' '.join(detected_text)
        return full_text if full_text else None
    
    def speak_text(self, text):
        """Metni sesli oku"""
        if not self.speaking:
            self.speaking = True
            try:
                print(f"🔊 Okunan: {text}")
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"Ses hatası: {e}")
            finally:
                self.speaking = False
    
    def speech_worker(self):
        """Arka planda ses okuma"""
        while True:
            text = self.text_queue.get()
            if text is None:
                break
            self.speak_text(text)
            self.text_queue.task_done()
    
    def is_duplicate(self, text):
        """Benzer metin kontrolü"""
        for prev_text in self.detected_texts:
            # Kelime bazlı benzerlik kontrolü
            similarity = len(set(text.split()) & set(prev_text.split())) / \
                        max(len(set(text.split())), len(set(prev_text.split())), 1)
            if similarity > 0.7:
                return True
        return False
    
    def process_image(self, image_path):
        """Resim dosyası işle"""
        print(f"\n📷 Resim işleniyor: {image_path}")
        
        frame = cv2.imread(image_path)
        if frame is None:
            print("❌ Resim yüklenemedi!")
            return
        
        # OCR yap
        ocr_data = self.detect_text_with_boxes(frame)
        text = self.draw_boxes_and_text(frame, ocr_data)
        
        if text:
            print(f"✅ Tespit edilen metin: {text}")
            self.speak_text(text)
        else:
            print("⚠️ Metin tespit edilemedi!")
        
        # Sonucu göster
        cv2.imshow('OCR Sonucu', frame)
        print("\n⌨️  Herhangi bir tuşa basın...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def process_video(self, video_path):
        """Video dosyası işle"""
        print(f"\n🎬 Video işleniyor: {video_path}")
        
        # Ses thread başlat
        speech_thread = threading.Thread(target=self.speech_worker, daemon=True)
        speech_thread.start()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Video açılamadı!")
            return
        
        fps_time = time.time()
        fps_counter = 0
        current_fps = 0
        
        print("⌨️  'q' = Çıkış, 'SPACE' = Ekran görüntüsü\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("✅ Video bitti!")
                break
            
            self.frame_count += 1
            fps_counter += 1
            
            # FPS hesapla
            if time.time() - fps_time > 1.0:
                current_fps = fps_counter / (time.time() - fps_time)
                fps_counter = 0
                fps_time = time.time()
            
            # FPS göster
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
            # OCR işle (Frame atlayarak)
            if self.frame_count % self.frame_skip == 0 and not self.speaking:
                ocr_data = self.detect_text_with_boxes(frame)
                text = self.draw_boxes_and_text(frame, ocr_data)
                
                if text and not self.is_duplicate(text):
                    self.detected_texts.append(text)
                    self.text_queue.put(text)
            
            # Durum
            status = "🔊 Konuşuyor" if self.speaking else "👁️ Taranıyor"
            cv2.putText(frame, status, (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            cv2.imshow('Video OCR (CPU)', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                filename = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Kaydedildi: {filename}")
        
        self.text_queue.put(None)
        cap.release()
        cv2.destroyAllWindows()
    
    def process_camera(self, camera_id=0):
        """Kamera canlı işle"""
        print(f"\n📹 Kamera başlatılıyor... (ID: {camera_id})")
        
        # Ses thread başlat
        speech_thread = threading.Thread(target=self.speech_worker, daemon=True)
        speech_thread.start()
        
        cap = cv2.VideoCapture(camera_id)
        
        # Kamera ayarları
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("❌ Kamera açılamadı!")
            return
        
        fps_time = time.time()
        fps_counter = 0
        current_fps = 0
        
        print("⌨️  'q' = Çıkış, 'SPACE' = Ekran görüntüsü\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame alınamadı!")
                break
            
            self.frame_count += 1
            fps_counter += 1
            
            # FPS hesapla
            if time.time() - fps_time > 1.0:
                current_fps = fps_counter / (time.time() - fps_time)
                fps_counter = 0
                fps_time = time.time()
            
            # FPS göster
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
            # OCR işle
            if self.frame_count % self.frame_skip == 0 and not self.speaking:
                ocr_data = self.detect_text_with_boxes(frame)
                text = self.draw_boxes_and_text(frame, ocr_data)
                
                if text and not self.is_duplicate(text):
                    self.detected_texts.append(text)
                    self.text_queue.put(text)
            
            # Durum
            status = "🔊 Konuşuyor" if self.speaking else "👁️ Taranıyor"
            cv2.putText(frame, status, (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            cv2.imshow('Kamera OCR (CPU)', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                filename = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Kaydedildi: {filename}")
        
        self.text_queue.put(None)
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Ana program - Kullanıcı girişi"""
    print("\n" + "="*60)
    print("🎯 EVRENSEL OCR OKUYUCU - Resim/Video/Kamera (CPU)")
    print("="*60)
    
    # Doğrudan başlat (CUDA sorma yok)
    reader = UniversalOCRReader()
    
    # Mod seçimi
    print("\n📝 Mod Seçin:")
    print("  1 - 📷 Resim dosyası")
    print("  2 - 🎬 Video dosyası")
    print("  3 - 📹 Kamera (canlı)")
    
    choice = input("\nSeçim (1/2/3): ").strip()
    
    if choice == '1':
        path = input("📷 Resim dosya yolu: ").strip()
        if os.path.exists(path):
            reader.process_image(path)
        else:
            print("❌ Dosya bulunamadı!")
    
    elif choice == '2':
        path = input("🎬 Video dosya yolu: ").strip()
        if os.path.exists(path):
            reader.process_video(path)
        else:
            print("❌ Dosya bulunamadı!")
    
    elif choice == '3':
        camera_id = input("📹 Kamera ID (varsayılan 0): ").strip()
        camera_id = int(camera_id) if camera_id else 0
        reader.process_camera(camera_id)
    
    else:
        print("❌ Geçersiz seçim!")

if __name__ == "__main__":
    main()