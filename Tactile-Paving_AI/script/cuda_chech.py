import subprocess
import sys
import os

def run_command(command):
    """Komutu çalıştır ve çıktıyı döndür"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception as e:
        return None

def check_cuda():
    """CUDA sürümlerini kontrol et"""
    print("=" * 60)
    print("🔍 CUDA VE GPU KONTROL ARACI")
    print("=" * 60)
    
    # 1. nvidia-smi kontrolü
    print("\n📊 1. NVIDIA Driver ve Runtime CUDA:")
    print("-" * 60)
    nvidia_smi = run_command("nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader")
    if nvidia_smi:
        print(f"   ✅ nvidia-smi çalışıyor")
        lines = nvidia_smi.split('\n')
        for line in lines:
            print(f"   {line}")
        
        # CUDA version from nvidia-smi
        cuda_version = run_command("nvidia-smi | grep 'CUDA Version' | awk '{print $9}'")
        if cuda_version:
            print(f"   🎯 Runtime CUDA Version: {cuda_version}")
    else:
        print("   ❌ nvidia-smi bulunamadı veya çalışmıyor")
    
    # 2. nvcc kontrolü (Compiler CUDA)
    print("\n🔧 2. CUDA Compiler (nvcc):")
    print("-" * 60)
    nvcc_version = run_command("nvcc --version | grep 'release' | awk '{print $5}' | cut -c2-")
    if nvcc_version:
        print(f"   ✅ nvcc bulundu")
        print(f"   🎯 Compiler CUDA Version: {nvcc_version}")
    else:
        print("   ❌ nvcc bulunamadı (CUDA Toolkit yüklü değil olabilir)")
    
    # 3. CUDA path kontrolü
    print("\n📁 3. CUDA Yol Kontrolleri:")
    print("-" * 60)
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home:
        print(f"   ✅ CUDA_HOME: {cuda_home}")
    else:
        print("   ⚠️  CUDA_HOME environment variable tanımlı değil")
    
    # Standart CUDA yollarını kontrol et
    cuda_paths = [
        '/usr/local/cuda',
        '/usr/local/cuda-12.8',
        '/usr/local/cuda-12.4',
        '/usr/local/cuda-12.1',
        '/usr/local/cuda-11.8',
        '/usr/local/cuda-11.7',
    ]
    
    found_paths = []
    for path in cuda_paths:
        if os.path.exists(path):
            found_paths.append(path)
    
    if found_paths:
        print(f"   ✅ Bulunan CUDA yolları:")
        for path in found_paths:
            print(f"      - {path}")
    else:
        print("   ⚠️  Standart CUDA yollarında CUDA bulunamadı")
    
    # 4. PyTorch CUDA kontrolü
    print("\n🐍 4. PyTorch CUDA Durumu:")
    print("-" * 60)
    try:
        import torch
        print(f"   ✅ PyTorch Version: {torch.__version__}")
        print(f"   🎯 CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   ✅ CUDA Version (PyTorch): {torch.version.cuda}")
            print(f"   ✅ cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"   ✅ GPU Count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"\n   📱 GPU {i}:")
                print(f"      - Name: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"      - Memory: {props.total_memory / 1024**3:.2f} GB")
                print(f"      - Compute Capability: {props.major}.{props.minor}")
        else:
            print("   ❌ PyTorch CUDA kullanılamıyor!")
            print("\n   💡 Olası sebepler:")
            print("      1. PyTorch CPU versiyonu yüklü")
            print("      2. CUDA sürümü PyTorch ile uyumsuz")
            print("      3. NVIDIA driver sorunu")
            
    except ImportError:
        print("   ❌ PyTorch yüklü değil")
    
    # 5. LD_LIBRARY_PATH kontrolü
    print("\n📚 5. Library Path:")
    print("-" * 60)
    ld_path = os.environ.get('LD_LIBRARY_PATH')
    if ld_path:
        cuda_in_path = any('cuda' in p.lower() for p in ld_path.split(':'))
        if cuda_in_path:
            print(f"   ✅ LD_LIBRARY_PATH'te CUDA var")
        else:
            print(f"   ⚠️  LD_LIBRARY_PATH'te CUDA yok")
    else:
        print("   ⚠️  LD_LIBRARY_PATH tanımlı değil")
    
    # 6. Özet ve Öneriler
    print("\n" + "=" * 60)
    print("📋 ÖZET VE ÖNERİLER")
    print("=" * 60)
    
    if nvidia_smi and nvcc_version:
        print("✅ CUDA donanım ve yazılım desteği mevcut")
        print(f"✅ Runtime CUDA: {cuda_version if cuda_version else 'Tespit edilemedi'}")
        print(f"✅ Compiler CUDA: {nvcc_version}")
    elif nvidia_smi:
        print("⚠️  NVIDIA GPU var ama CUDA Toolkit eksik olabilir")
        print("💡 CUDA Toolkit yükleyin: https://developer.nvidia.com/cuda-downloads")
    else:
        print("❌ NVIDIA GPU veya driver tespit edilemedi")
    
    try:
        import torch
        if not torch.cuda.is_available():
            print("\n❌ PyTorch CUDA kullanamıyor!")
            print("\n🔧 ÇÖZÜMLERİ:")
            if cuda_version:
                major_version = cuda_version.split('.')[0]
                print(f"\n   1. PyTorch'u CUDA {cuda_version} için yükleyin:")
                if major_version in ['12']:
                    print(f"      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
                elif major_version in ['11']:
                    print(f"      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            else:
                print("\n   1. PyTorch'u CUDA destekli yükleyin:")
                print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            
            print("\n   2. Veya conda ile:")
            print("      conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia")
    except:
        pass
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    check_cuda()