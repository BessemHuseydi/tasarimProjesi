import cv2
import time
import os
from google import genai
from PIL import Image
from gtts import gTTS
from playsound import playsound

# --- AYARLAR ---
# 1. API ANAHTARINI BURAYA YAZ
API_KEY = "AIzaSyBhmyxmrXGvOQzVGXsOopM5UnrOsLenXD4"

# 2. KAMERA AYARI (Camo Studio İçin)
# Eğer çalışmazsa burayı 1 yap:
KAMERA_PORTU = 0

# İstemciyi başlatıyoruz
try:
    client = genai.Client(api_key=API_KEY)
except Exception as e:
    print(f"Anahtar hatası: {e}")

def seslendir(metin):
    """Metni okur ve dosyayı hemen siler."""
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

        prompt = "Bu resimde ne var? Görme engelli biri için ortamı ve tehlikeleri Türkçe olarak tek cümleyle özetle."

        # Hızlı ve ücretsiz model
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=[pil_image, prompt]
        )
        return response.text
        
    except Exception as e:
        print(f"🛑 API Hatası: {e}")
        if "429" in str(e):
            return "Çok hızlı işlem yapıldı, biraz bekle."
        return "Bağlantı sorunu oluştu."

def main():
    # --- KRİTİK NOKTA: CAMO STUDIO AYARI ---
    # cv2.CAP_DSHOW komutu, Windows'ta Camo'nun görünmesini sağlar.
    cap = cv2.VideoCapture(KAMERA_PORTU, cv2.CAP_DSHOW)

    # Çözünürlüğü HD yapalım (Camo destekler)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"Kamera (Port {KAMERA_PORTU}) açılamadı! Port numarasını değiştirmeyi dene.")
        return

    print(f"--- GÖRME ENGELLİ ASİSTANI (Camo Modu: Port {KAMERA_PORTU}) ---")
    print("Program hazır. Fotoğraf çekmek için 's' tuşuna bas.")
    print("Çıkmak için 'q' tuşuna bas.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Görüntü alınamadı. Camo Studio açık mı?")
            break

        cv2.imshow('Kamera (Camo Studio)', frame)
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


# import cv2
# import time
# import os
# from google import genai
# from PIL import Image
# import pyttsx3  # Ses kütüphanesi

# # --- AYARLAR ---
# API_KEY = "AIzaSyCH2jCqY2VVwnDEpunXseC17k9Y__IoMS8"
# KAMERA_PORTU = 0  # Camo için 0 veya 1 olabilir

# # --- SES AYARLARI (TÜRKÇE SEÇİMİ) ---
# engine = pyttsx3.init()

# # Hız ayarı (200-230 arası iyidir)
# engine.setProperty('rate', 100)

# # SİSTEMDEKİ TÜRKÇE SESİ BULMA
# voices = engine.getProperty('voices')
# turkce_ses_bulundu = False

# for voice in voices:
#     # Sesin adında veya ID'sinde 'turkish' veya 'tr' geçiyor mu bak
#     if "turkish" in voice.name.lower() or "tr-" in voice.id.lower():
#         engine.setProperty('voice', voice.id)
#         print(f"✅ Türkçe ses seçildi: {voice.name}")
#         turkce_ses_bulundu = True
#         break

# if not turkce_ses_bulundu:
#     print("⚠️ UYARI: Bilgisayarda Türkçe ses paketi bulunamadı!")
#     print("Ses İngilizce aksanıyla çıkabilir. Windows Ayarları > Zaman ve Dil > Konuşma kısmından Türkçe'yi yüklemelisin.")

# # İstemciyi başlat
# try:
#     client = genai.Client(api_key=API_KEY)
# except Exception as e:
#     print(f"Anahtar hatası: {e}")

# def seslendir(metin):
#     """Metni sesli okur."""
#     if not metin: return

#     print(f"🤖 Asistan: {metin}")
#     try:
#         engine.say(metin)
#         engine.runAndWait()
#     except Exception as e:
#         print(f"Ses hatası: {e}")

# def resim_analiz_et(cv2_resim):
#     print("⏳ Resim analiz ediliyor...")
#     try:
#         img_rgb = cv2.cvtColor(cv2_resim, cv2.COLOR_BGR2RGB)
#         pil_image = Image.fromarray(img_rgb)

#         prompt = "Bu resimde ne var? Görme engelli biri için tek cümleyle net bir şekilde anlat."

#         response = client.models.generate_content(
#             model="gemini-flash-latest",
#             contents=[pil_image, prompt]
#         )
#         return response.text
        
#     except Exception as e:
#         print(f"Hata: {e}")
#         return "Bağlantı sorunu."

# def main():
#     cap = cv2.VideoCapture(KAMERA_PORTU, cv2.CAP_DSHOW)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#     if not cap.isOpened():
#         print(f"Kamera (Port {KAMERA_PORTU}) açılamadı! Port numarasını 1 yapmayı dene.")
#         return

#     print("--- GÖRME ENGELLİ ASİSTANI (HIZLI & TÜRKÇE) ---")
#     print("Çekmek için 's', Çıkmak için 'q' tuşuna bas.")

#     while True:
#         ret, frame = cap.read()
#         if not ret: break

#         cv2.imshow('Kamera', frame)
#         key = cv2.waitKey(1) & 0xFF

#         if key == ord('s'):
#             print("\n📸 İşleniyor...")
#             aciklama = resim_analiz_et(frame)
#             seslendir(aciklama)

#         elif key == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()