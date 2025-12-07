import os
from pathlib import Path

# Dataset yolu (gerekirse değiştir)
base_dir = Path(
    r"C:/Users/besse/coding/tasarim/Tactile-Paving_AI/data/tactile paving image dataset.v3i.yolov8"
)

splits = ["train", "valid"]   # train / val

deleted_labels = 0
deleted_images = 0

for split in splits:
    labels_dir = base_dir / split / "labels"
    images_dir = base_dir / split / "images"

    if not labels_dir.exists():
        print(f"⚠️ {labels_dir} yok, atlandı")
        continue

    for label_file in labels_dir.glob("*.txt"):
        with open(label_file, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        contains_box = False

        for line in lines:
            parts = line.split()
            if len(parts) == 5:   # class cx cy w h
                contains_box = True
                break

        if contains_box:
            # Label sil
            label_file.unlink()
            deleted_labels += 1

            # Image sil
            for ext in [".jpg", ".png", ".jpeg"]:
                img_path = images_dir / (label_file.stem + ext)
                if img_path.exists():
                    img_path.unlink()
                    deleted_images += 1
                    break

print("✅ Temizlik tamamlandı")
print(f"🗑️ Silinen label: {deleted_labels}")
print(f"🗑️ Silinen image: {deleted_images}")
for split in ["train", "valid"]:
    labels_dir = base_dir / split / "labels"
    for f in labels_dir.glob("*.txt"):
        remove = False
        with open(f, "r") as file:
            for line in file:
                if len(line.strip().split()) <= 5:
                    remove = True
                    break
        if remove:
            print("🗑️ Silindi:", f.name)
            f.unlink()  # Dosyayı sil
            # İlgili görseli de sil (aynı ada sahip .jpg veya .png)
            image_path_jpg = f.with_suffix('.jpg').as_posix().replace("/labels/", "/images/")
            image_path_png = f.with_suffix('.png').as_posix().replace("/labels/", "/images/")
            Path(image_path_jpg).unlink(missing_ok=True)
            Path(image_path_png).unlink(missing_ok=True)
