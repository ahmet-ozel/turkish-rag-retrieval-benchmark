# Production Deployment Kılavuzu

## 🚀 Neden FastAPI?

Bu uygulama için FastAPI seçilmesinin nedenleri:

1. **CPU-Tabanlı İşlemler**: GPU gerektirmeyen metin işleme
2. **Async I/O**: Dosya okuma/yazma için verimli
3. **Yüksek Performans**: Starlette ve Pydantic üzerine kurulu
4. **Otomatik Dokümantasyon**: Swagger/OpenAPI desteği
5. **Type Safety**: Pydantic ile veri doğrulama

## 📊 Performans Optimizasyonları

### 1. Multi-Process Deployment

```bash
# CPU çekirdek sayısına göre worker ayarla
gunicorn main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 300 \
  --keep-alive 5 \
  --max-requests 1000 \
  --max-requests-jitter 50
```

### 2. Async Task Queue (Büyük dosyalar için)

Celery entegrasyonu örneği:

```python
# tasks.py
from celery import Celery
import asyncio

celery_app = Celery('chunking', broker='redis://localhost:6379')

@celery_app.task
def process_large_file(file_path, params):
    # Chunking işlemi
    return results
```

### 3. Connection Pooling

```python
# PostgreSQL için
from databases import Database
database = Database('postgresql://user:password@localhost/db', min_size=10, max_size=20)

# MongoDB için
from motor.motor_asyncio import AsyncIOMotorClient
client = AsyncIOMotorClient('mongodb://localhost:27017', maxPoolSize=50)
```

## 🔧 Sistem Optimizasyonları

### Linux Kernel Tuning

```bash
# /etc/sysctl.conf
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.ipv4.ip_local_port_range = 1024 65535
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_fin_timeout = 15
fs.file-max = 100000
```

### Nginx Optimizasyonları

```nginx
# nginx.conf
worker_processes auto;
worker_rlimit_nofile 65535;

events {
    worker_connections 4096;
    use epoll;
    multi_accept on;
}

http {
    # Caching
    proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=api_cache:10m max_size=1g;
    
    # Compression
    gzip on;
    gzip_comp_level 6;
    gzip_types application/json text/plain;
    
    # Keep-alive
    keepalive_requests 100;
    keepalive_timeout 65;
}
```

## 📈 Monitoring Stack

### 1. Prometheus + Grafana

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

### 2. Application Metrics

```python
# main.py'ye ekle
from prometheus_fastapi_instrumentator import Instrumentator

@app.on_event("startup")
async def startup():
    Instrumentator().instrument(app).expose(app)
```

## 🛡️ Güvenlik Önlemleri

### 1. Rate Limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/api/v1/process")
@limiter.limit("5/minute")
async def process_document(...):
    pass
```

### 2. Input Validation

```python
# Dosya tipi ve boyut kontrolü
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'csv', 'xlsx'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def validate_file(file: UploadFile):
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(400, "Dosya çok büyük")
    
    ext = file.filename.split('.')[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Geçersiz dosya tipi")
```

## 🏗️ Deployment Stratejileri

### 1. Blue-Green Deployment

```bash
# Yeni version'ı deploy et
docker build -t chunking-api:v2 .
docker run -d --name api-green -p 8001:8000 chunking-api:v2

# Test et
curl http://localhost:8001/health

# Traffic'i yönlendir
# nginx.conf'ta upstream'i güncelle

# Eski version'ı kaldır
docker stop api-blue && docker rm api-blue
```

### 2. Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: chunking-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: chunking-api
  template:
    metadata:
      labels:
        app: chunking-api
    spec:
      containers:
      - name: api
        image: chunking-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

## 📊 Benchmark Sonuçları

Tipik performans değerleri (4 CPU, 8GB RAM):

| Dosya Tipi | Boyut | İşlem Süresi | RPS |
|------------|-------|--------------|-----|
| TXT | 1MB | ~100ms | 40 |
| PDF | 5MB | ~500ms | 8 |
| DOCX | 2MB | ~200ms | 20 |
| CSV | 10MB | ~1s | 4 |

## 🔍 Debugging ve Profiling

### 1. Performance Profiling

```python
import cProfile
import pstats

def profile_endpoint():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # İşlem
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
```

### 2. Memory Profiling

```python
from memory_profiler import profile

@profile
def process_large_file():
    # Memory-intensive işlemler
    pass
```

## 💡 Best Practices

1. **Connection Pooling**: Veritabanı bağlantıları için pool kullanın
2. **Caching**: Sık kullanılan sonuçları Redis'te cache'leyin
3. **Async Everything**: Mümkün olan her yerde async kullanın
4. **Batch Processing**: Toplu işlemler için batch endpoint'leri kullanın
5. **Health Checks**: Düzenli sağlık kontrolleri yapın
6. **Logging**: Structured logging kullanın (JSON format)
7. **Error Handling**: Global exception handler kullanın
8. **Documentation**: API dokümantasyonunu güncel tutun

## 🚨 Sorun Giderme

### High CPU Usage
- Worker sayısını azaltın
- Chunk boyutunu optimize edin
- CPU profiling yapın

### Memory Leaks
- Max requests per worker ayarlayın
- Memory profiling yapın
- Garbage collection'ı tune edin

### Slow Response Times
- Async endpoints kullanın
- Database query'leri optimize edin
- Caching ekleyin

### Connection Errors
- Connection pool boyutunu artırın
- Timeout değerlerini ayarlayın
- Keep-alive ayarlarını kontrol edin
