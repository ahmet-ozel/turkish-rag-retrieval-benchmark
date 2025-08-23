"""
API Test Script
Bu script ile API'nin çalışıp çalışmadığını test edebilirsiniz.
"""

import requests
import json
import time
import sys
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Sağlık kontrolü"""
    print("🏥 Sağlık kontrolü yapılıyor...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ API sağlıklı!")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ API yanıt vermiyor! Status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API'ye bağlanılamıyor! Hata: {e}")
        return False

def test_sync_processing():
    """Senkron dosya işleme testi"""
    print("\n📄 Senkron işleme testi...")
    
    # Test dosyası oluştur
    test_file = "test_document.txt"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("""Bu bir test dokümanıdır.

İlk paragraf burada yer alıyor. Bu paragraf chunking işlemi için yeterince uzun olmalıdır.

İkinci paragraf daha uzun bir metin içeriyor. FastAPI ile oluşturduğumuz API'nin chunking özelliklerini test ediyoruz. Bu metin farklı boyutlarda chunk'lara bölünecek.

Üçüncü paragraf da test için eklendi. Her paragraf ayrı bir chunk olabilir veya birden fazla paragraf tek bir chunk'ta birleştirilebilir. Bu tamamen seçilen yönteme ve parametrelere bağlıdır.

Son paragraf olarak, bu test dokümanının amacı API'nin düzgün çalıştığını doğrulamaktır.""")
    
    try:
        with open(test_file, "rb") as f:
            files = {"file": (test_file, f, "text/plain")}
            params = {
                "method": "Ayırıcı Bazlı",
                "chunk_size": 200,
                "separator": "\n\n"
            }
            
            response = requests.post(
                f"{BASE_URL}/api/v1/process-sync",
                files=files,
                params=params
            )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Senkron işleme başarılı!")
            print(f"   Toplam chunk: {result['total_chunks']}")
            print(f"   Ortalama boyut: {result['statistics']['average_chunk_size']:.0f} karakter")
            
            # İlk chunk'ı göster
            if result['chunks']:
                print(f"\n   İlk chunk örneği:")
                print(f"   {result['chunks'][0]['text'][:100]}...")
            return True
        else:
            print(f"❌ İşleme hatası! Status: {response.status_code}")
            print(f"   Hata: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test başarısız! Hata: {e}")
        return False
    finally:
        # Test dosyasını temizle
        if Path(test_file).exists():
            Path(test_file).unlink()

def test_async_processing():
    """Asenkron dosya işleme testi"""
    print("\n⚡ Asenkron işleme testi...")
    
    # Büyük test dosyası oluştur
    test_file = "test_large_document.txt"
    with open(test_file, "w", encoding="utf-8") as f:
        for i in range(100):
            f.write(f"Paragraf {i+1}: Bu paragraf asenkron işleme testini yapmak için oluşturuldu. " * 5)
            f.write("\n\n")
    
    try:
        # Dosyayı yükle ve işleme başlat
        with open(test_file, "rb") as f:
            files = {"file": (test_file, f, "text/plain")}
            params = {
                "method": "Sabit Boyut",
                "chunk_size": 500,
                "chunk_overlap": 50
            }
            
            response = requests.post(
                f"{BASE_URL}/api/v1/process",
                files=files,
                params=params
            )
        
        if response.status_code == 200:
            result = response.json()
            job_id = result['job_id']
            print(f"✅ İşlem başlatıldı! Job ID: {job_id}")
            
            # İşlem durumunu kontrol et
            print("   İşlem durumu kontrol ediliyor", end="")
            for i in range(30):  # Max 30 saniye bekle
                time.sleep(1)
                print(".", end="", flush=True)
                
                status_response = requests.get(f"{BASE_URL}/api/v1/status/{job_id}")
                if status_response.status_code == 200:
                    status = status_response.json()
                    
                    if status['status'] == 'completed':
                        print("\n✅ İşlem tamamlandı!")
                        print(f"   Toplam chunk: {status['result']['total_chunks']}")
                        print(f"   Toplam karakter: {status['result']['statistics']['total_characters']:,}")
                        return True
                    elif status['status'] == 'error':
                        print(f"\n❌ İşlem hatası: {status['error']}")
                        return False
            
            print("\n⏱️  İşlem zaman aşımına uğradı!")
            return False
            
        else:
            print(f"❌ İşlem başlatılamadı! Status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Test başarısız! Hata: {e}")
        return False
    finally:
        # Test dosyasını temizle
        if Path(test_file).exists():
            Path(test_file).unlink()

def test_batch_processing():
    """Toplu dosya işleme testi"""
    print("\n📦 Toplu işleme testi...")
    
    # Birden fazla test dosyası oluştur
    test_files = []
    for i in range(3):
        filename = f"test_batch_{i+1}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Dosya {i+1} içeriği\n\n")
            f.write("Bu dosya toplu işleme testinin bir parçasıdır.\n" * 10)
        test_files.append(filename)
    
    try:
        # Dosyaları yükle
        files = []
        for filename in test_files:
            files.append(('files', (filename, open(filename, 'rb'), 'text/plain')))
        
        params = {
            "method": "Sabit Boyut",
            "chunk_size": 300
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/batch-process",
            files=files,
            params=params
        )
        
        # Dosyaları kapat
        for _, (_, f, _) in files:
            f.close()
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Toplu işlem başlatıldı!")
            print(f"   İşlenen dosya sayısı: {len(result['jobs'])}")
            
            # Her job'ın durumunu kontrol et
            all_completed = True
            for job in result['jobs']:
                if job.get('error'):
                    print(f"   ❌ {job['filename']}: {job['error']}")
                    all_completed = False
                else:
                    print(f"   ✅ {job['filename']}: Job ID {job['job_id']}")
            
            return all_completed
        else:
            print(f"❌ Toplu işlem başlatılamadı! Status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Test başarısız! Hata: {e}")
        return False
    finally:
        # Test dosyalarını temizle
        for filename in test_files:
            if Path(filename).exists():
                Path(filename).unlink()

def main():
    """Ana test fonksiyonu"""
    print("🧪 Doküman Chunking API Test Script")
    print("=" * 50)
    
    # API'nin çalıştığını kontrol et
    if not test_health():
        print("\n⚠️  API çalışmıyor! Lütfen önce API'yi başlatın:")
        print("   uvicorn main:app --reload")
        sys.exit(1)
    
    # Testleri çalıştır
    tests = [
        test_sync_processing,
        test_async_processing,
        test_batch_processing
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test çalıştırılırken hata: {e}")
            failed += 1
    
    # Sonuçları göster
    print("\n" + "=" * 50)
    print("📊 Test Sonuçları:")
    print(f"   ✅ Başarılı: {passed}")
    print(f"   ❌ Başarısız: {failed}")
    print(f"   📈 Başarı oranı: {(passed/(passed+failed)*100):.0f}%")
    
    if failed == 0:
        print("\n🎉 Tüm testler başarılı! API kullanıma hazır.")
    else:
        print("\n⚠️  Bazı testler başarısız oldu. Lütfen hataları kontrol edin.")

if __name__ == "__main__":
    main()
