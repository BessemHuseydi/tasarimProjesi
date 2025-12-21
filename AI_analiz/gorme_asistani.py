import cv2
import time
import os
from google import genai
from PIL import Image
from gtts import gTTS
from playsound import playsound

# --- AYARLAR ---
# API ANAHTARINI BURAYA YAZ
API_KEY = "AIzaSyBc1EjuUoTLHP9sJwtHPYE3rvI-9YLqGLw"

# İstemciyi başlatıyoruz
try:
    client = genai.Client(api_key=API_KEY)
except Exception as e:
    print(f"Anahtar hatası: {e}")

def seslendir(metin):
    """Metni okur ve dosyayı hemen siler (Kayıt tutmaz)."""
    if not metin: return

    print(f"🤖 Asistan: {metin}")
    dosya_adi = f"ses_{int(time.time())}.mp3"
    
    try:
        tts = gTTS(text=metin, lang='tr')
        tts.save(dosya_adi)
        playsound(dosya_adi)
    except Exception as e:
        print(f"Ses hatası: {e}")
    finally:
        if os.path.exists(dosya_adi):
            try:
                os.remove(dosya_adi)
            except:
                pass 

def resim_analiz_et(cv2_resim):
    """Resmi Gemini'ye gönderir."""
    print("⏳ Resim analiz ediliyor...")
    
    try:
        img_rgb = cv2.cvtColor(cv2_resim, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)

        prompt = "Bu resimde ne var? Görme engelli biri için ortamı ve tehlikeleri Türkçe olarak kısaca anlat."

        # DEĞİŞİKLİK BURADA:
        # Kotası en yüksek olan 'gemini-flash-latest' modelini kullanıyoruz.
        # Bu model senin listende mevcuttu ve ücretsiz kullanım için en iyisidir.
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=[pil_image, prompt]
        )
        
        return response.text
        
    except Exception as e:
        print(f"🛑 API Hatası: {e}")
        return "Bağlantı sorunu oluştu."

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Kamera açılamadı!")
        return

    print("--- GÖRME ENGELLİ ASİSTANI (FLASH LATEST) ---")
    print("Program hazır. Fotoğraf çekmek için 's' tuşuna bas.")
    print("Çıkmak için 'q' tuşuna bas.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Görüntü alınamadı.")
            break

        cv2.imshow('Kamera', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            print("\n📸 Fotoğraf çekildi, işleniyor...")
            aciklama = resim_analiz_et(frame)
            seslendir(aciklama)

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()