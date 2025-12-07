import os

def clean_yolo_dataset(dataset_dir, modes=["images", "labels"], seg=False):
    """
    YOLO dataset temizleme fonksiyonu.
    Detection ve Segmentasyon için uyumsuz dosyaları otomatik siler.

    dataset_dir: train veya val klasör yolu
    modes: ["images","labels"] - temizlenecek kategoriler
    seg: True ise segmentation mask kontrolü yapılır
    """

    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")

    deleted_images = []
    deleted_labels = []

    # 1. Etiketi olmayan resimleri sil
    for img in os.listdir(images_dir):
        if img.lower().endswith((".jpg", ".jpeg", ".png")):
            name = os.path.splitext(img)[0]
            label_path = os.path.join(labels_dir, name + ".txt")

            if not os.path.exists(label_path):
                os.remove(os.path.join(images_dir, img))
                deleted_images.append(img)

    # 2. Resmi olmayan etiketleri sil
    for lbl in os.listdir(labels_dir):
        if lbl.endswith(".txt"):
            name = os.path.splitext(lbl)[0]
            img_exists = any(os.path.exists(os.path.join(images_dir, name + ext))
                             for ext in [".jpg", ".jpeg", ".png"])

            if not img_exists:
                os.remove(os.path.join(labels_dir, lbl))
                deleted_labels.append(lbl)

    # 3. Segmentasyon için mask kontrolü
    if seg:
        for lbl in os.listdir(labels_dir):
            label_path = os.path.join(labels_dir, lbl)
            with open(label_path, "r") as f:
                lines = f.readlines()

            delete_flag = False

            for line in lines:
                parts = line.strip().split()

                # YOLO segmentation format: class x1 y1 x2 y2 ... (min 4 koordinat)
                if len(parts) < 6:  
                    delete_flag = True
                    break

            if delete_flag:
                name = os.path.splitext(lbl)[0]
                # etiketi sil
                os.remove(label_path)
                deleted_labels.append(lbl)

                # resmi sil
                for ext in [".jpg", ".jpeg", ".png"]:
                    img_path = os.path.join(images_dir, name + ext)
                    if os.path.exists(img_path):
                        os.remove(img_path)
                        deleted_images.append(name + ext)

    print("\n=== Silinen Resimler ===")
    print(deleted_images)
    print("\n=== Silinen Etiketler ===")
    print(deleted_labels)
    print("\nTemizleme işlemi tamamlandı:", dataset_dir)



# ----------------------------------------------------------------
# 2. TRAIN + VAL KLASÖRLERİNİ OTOMATİK TEMİZLE
# ----------------------------------------------------------------

root = r"C:\Users\besse\coding\tasarim\Tactile-Paving_AI\data\tactile_dataset"

train_dir = os.path.join(root, "train")
val_dir = os.path.join(root, "val")

print("\n--- TRAIN SET CLEANING ---")
clean_yolo_dataset(train_dir, seg=True)

print("\n--- VAL SET CLEANING ---")
clean_yolo_dataset(val_dir, seg=True)
