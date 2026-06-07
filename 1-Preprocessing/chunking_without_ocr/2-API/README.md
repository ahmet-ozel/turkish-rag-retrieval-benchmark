# Doküman Chunking API

Bu API, dokümanları anlamlı parçalara (chunk) bölen bir servistir. PDF, DOCX, TXT, CSV ve Excel dosyalarını destekler.

## Hızlı Başlangıç

### 1. Gereksinimler
- Python 3.11+
- pip

### 2. Kurulum

```bash
# Bağımlılıkları yükle
pip install -r requirements.txt

# API'yi başlat
uvicorn main:app --reload
```

API şu adreste çalışacaktır: http://localhost:8000

## API Dokümantasyonu

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Temel Kullanım

### Dosya İşleme (Async)
```bash
curl -X POST "http://localhost:8000/api/v1/process" \
  -H "accept: application/json" \
  -F "file=@dokuman.pdf" \
  -F "method=Sabit Boyut" \
  -F "chunk_size=500"
```

### İşlem Durumu Sorgulama
```bash
curl -X GET "http://localhost:8000/api/v1/status/{job_id}"
```

### Senkron İşleme (Küçük dosyalar için)
```bash
curl -X POST "http://localhost:8000/api/v1/process-sync" \
  -F "file=@dokuman.txt"
```

## Production Deployment

### Docker ile Deployment

```bash
# Image'ı build et
docker build -t chunking-api .

# Container'ı çalıştır
docker run -p 8000:8000 chunking-api
```

### Docker Compose ile Full Stack

```bash
# Tüm servisleri başlat (API + Redis + Nginx)
docker-compose up -d

# Logları izle
docker-compose logs -f
```

## Performans Optimizasyonları

### 1. Multi-Worker Deployment
```bash
# Gunicorn ile 4 worker
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### 2. Load Balancing
Nginx reverse proxy kullanarak yük dengeleme yapabilirsiniz. `nginx.conf` dosyasına bakın.

### 3. Caching (Opsiyonel)
Redis kullanarak sonuçları cache'leyebilirsiniz:

```python
# Redis entegrasyonu için main.py'yi güncelleyin
import redis
cache = redis.Redis(host='localhost', port=6379, db=0)
```

## Monitoring

### Prometheus Metrics
```python
# main.py'ye ekleyin
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)
```

Metrikler: http://localhost:8000/metrics

## Güvenlik

1. **Rate Limiting**: Nginx'te yapılandırılmıştır
2. **CORS**: Ayarlanabilir origin'ler
3. **File Size Limit**: Varsayılan 10MB
4. **Input Validation**: Pydantic modelleri ile

## Endpoint'ler

| Endpoint | Method | Açıklama |
|----------|--------|----------|
| `/` | GET | API bilgileri |
| `/health` | GET | Sağlık kontrolü |
| `/api/v1/process` | POST | Async dosya işleme |
| `/api/v1/process-sync` | POST | Senkron dosya işleme |
| `/api/v1/batch-process` | POST | Toplu dosya işleme |
| `/api/v1/status/{job_id}` | GET | İşlem durumu |
| `/api/v1/export/{job_id}` | GET | Sonuçları dışa aktar |

## Konfigürasyon

`config.env` dosyasını düzenleyerek ayarları değiştirebilirsiniz.

## İpuçları

1. **Büyük dosyalar için**: Async endpoint kullanın
2. **Toplu işlemler için**: Batch endpoint kullanın (max 10 dosya)
3. **CPU kullanımı**: Worker sayısını CPU çekirdek sayısına göre ayarlayın
4. **Memory kullanımı**: Chunk boyutunu ve overlap'i optimize edin

## Sorun Giderme

1. **Memory hatası**: Chunk boyutunu azaltın
2. **Timeout hatası**: Nginx/Gunicorn timeout değerlerini artırın
3. **Rate limit**: Rate limit değerlerini artırın veya Redis cache ekleyin
