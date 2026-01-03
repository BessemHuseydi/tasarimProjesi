# import google.generativeai as genai
# import cv2
# import PIL.Image
# import pyttsx3
# import time

# # 1. API Ayarları (BURAYA KENDİ ANAHTARINI GİR)
# API_KEY = "AIzaSyAuH3EvP2zlomcr5nTy5-RonaubBqqOQUk"
# genai.configure(api_key=API_KEY)

# # Modeli seçiyoruz (En kararlı ve ücretsiz model)
# model = genai.GenerativeModel('gemini-1.5-flash')

# # Ses motorunu başlat
# engine = pyttsx3.init()

# def sesli_uyari(metin):
#     print(f"Asistan: {metin}")
#     try:
#         engine.say(metin)
#         engine.runAndWait()
#     except:
#         pass

# def analiz_et_ve_yonlendir(frame):
#     # OpenCV (BGR) -> PIL (RGB) Dönüşümü
#     img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     img_pil = PIL.Image.fromarray(img_rgb)

#     prompt = """
#     Sen görme engelli birine rehberlik ediyorsun.
#     Fotoğrafa bak ve sadece yürüme yoluyla ilgili bilgi ver.
#     Önümde engel var mı? Nerede? (Sağda, solda, ortada).
#     Cevabın çok kısa ve net olsun. Örnek: "Önünde masa var, sağdan geç." veya "Yol açık."
#     """

#     try:
#         # Eski kütüphane kullanımı (Daha stabil)
#         response = model.generate_content([prompt, img_pil])
#         return response.text
#     except Exception as e:
#         print(f"HATA DETAYI: {e}") 
#         return "Bağlantı hatası."

# # Kamera Başlatma
# cap = cv2.VideoCapture(0)

# print("Sistem Hazır. Analiz için 'q' tuşuna basın. Çıkış için 'ESC'.")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Kamera açılmadı!")
#         break

#     cv2.imshow('Akilli Baston', frame)

#     key = cv2.waitKey(1) & 0xFF
    
#     # 'q' tuşuna basınca analiz et
#     if key == ord('q'): 
#         print("Sahne inceleniyor...")
#         sonuc = analiz_et_ve_yonlendir(frame)
#         sesli_uyari(sonuc)
    
#     # 'ESC' tuşu ile çık
#     if key == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()