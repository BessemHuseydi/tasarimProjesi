import os
import shutil
import yaml
from pathlib import Path
import random
from collections import defaultdict

# Kaynak dataset klasörleri
source_datasets = [
    "Tenji dots 2.v4i.yolov8",
    "Tenji from 10k.v1i.yolov8",
    "tenjima400..yolov8"
]

# Hedef klasör
output_dataset = "merged_tenji_dataset"

# Sınıf isimleri (data.yaml'dan okunacak)
class_names = ["line", "dot"]

def create_output_structure(output_path):
    """Çıktı klasör yapısını oluştur"""
    folders = ['train/images', 'train/labels', 
               'valid/images', 'valid/labels',
               'test/images', 'test/labels']
    
    for folder in folders:
        Path(os.path.join(output_path, folder)).mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Klasör yapısı oluşturuldu: {output_path}")

def collect_all_files(source_datasets):
    """Tüm veri setlerinden dosyaları topla"""
    all_files = []
    
    for dataset in source_datasets:
        for split in ['train', 'valid', 'test']:
            images_path = os.path.join(dataset, split, 'images')
            labels_path = os.path.join(dataset, split, 'labels')
            
            if os.path.exists(images_path):
                image_files = [f for f in os.listdir(images_path) 
                             if f.endswith(('.jpg', '.jpeg', '.png'))]
                
                for img_file in image_files:
                    label_file = os.path.splitext(img_file)[0] + '.txt'
                    img_path = os.path.join(images_path, img_file)
                    label_path = os.path.join(labels_path, label_file)
                    
                    if os.path.exists(label_path):
                        all_files.append({
                            'image': img_path,
                            'label': label_path,
                            'dataset': dataset
                        })
    
    print(f"✓ Toplam {len(all_files)} görsel-etiket çifti toplandı")
    return all_files

def split_data(all_files, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Veriyi 80-10-10 oranında böl"""
    random.shuffle(all_files)
    
    total = len(all_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    splits = {
        'train': all_files[:train_end],
        'valid': all_files[train_end:val_end],
        'test': all_files[val_end:]
    }
    
    print(f"✓ Veri bölündü: Train={len(splits['train'])}, Valid={len(splits['valid'])}, Test={len(splits['test'])}")
    return splits

def copy_files(splits, output_path):
    """Dosyaları yeni yapıya kopyala"""
    for split_name, files in splits.items():
        for idx, file_info in enumerate(files):
            # Benzersiz dosya adı oluştur
            ext = os.path.splitext(file_info['image'])[1]
            new_name = f"{split_name}_{idx:05d}{ext}"
            
            # Görsel kopyala
            src_img = file_info['image']
            dst_img = os.path.join(output_path, split_name, 'images', new_name)
            shutil.copy2(src_img, dst_img)
            
            # Etiket kopyala
            src_label = file_info['label']
            dst_label = os.path.join(output_path, split_name, 'labels', 
                                    os.path.splitext(new_name)[0] + '.txt')
            shutil.copy2(src_label, dst_label)
        
        print(f"✓ {split_name} dosyaları kopyalandı: {len(files)} adet")

def create_yaml_file(output_path, class_names):
    """data.yaml dosyasını oluştur"""
    yaml_content = {
        'path': os.path.abspath(output_path),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(class_names),
        'names': class_names
    }
    
    yaml_path = os.path.join(output_path, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✓ data.yaml dosyası oluşturuldu")

def main():
    print("=" * 50)
    print("YOLO Dataset Birleştirme ve Bölme")
    print("=" * 50)
    
    # Çıktı klasörünü temizle
    if os.path.exists(output_dataset):
        shutil.rmtree(output_dataset)
        print(f"✓ Eski klasör temizlendi")
    
    # 1. Klasör yapısını oluştur
    create_output_structure(output_dataset)
    
    # 2. Tüm dosyaları topla
    all_files = collect_all_files(source_datasets)
    
    if len(all_files) == 0:
        print("❌ Hiç dosya bulunamadı! Klasör yollarını kontrol edin.")
        return
    
    # 3. Veriyi böl
    splits = split_data(all_files, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    
    # 4. Dosyaları kopyala
    copy_files(splits, output_dataset)
    
    # 5. data.yaml oluştur
    create_yaml_file(output_dataset, class_names)
    
    print("=" * 50)
    print("✅ İşlem tamamlandı!")
    print(f"📁 Çıktı klasörü: {output_dataset}")
    print("=" * 50)

if __name__ == "__main__":
    main()