#!/bin/bash

# Doküman Chunking API Başlatma Script'i

echo "🚀 Doküman Chunking API Başlatılıyor..."
echo "======================================="

# Python versiyonunu kontrol et
python_version=$(python3 --version 2>&1)
echo "📍 Python versiyonu: $python_version"

# Virtual environment kontrol et veya oluştur
if [ ! -d "venv" ]; then
    echo "📦 Virtual environment oluşturuluyor..."
    python3 -m venv venv
fi

# Virtual environment'ı aktive et
echo "🔧 Virtual environment aktive ediliyor..."
source venv/bin/activate || . venv/Scripts/activate

# Bağımlılıkları yükle
echo "📚 Bağımlılıklar kontrol ediliyor..."
pip install -r requirements.txt --quiet

# Geçici dizini oluştur
mkdir -p /tmp

# Development veya Production modunu seç
if [ "$1" == "prod" ]; then
    echo "🏭 Production modunda başlatılıyor..."
    echo "   Workers: 4"
    echo "   Host: 0.0.0.0:8000"
    gunicorn main:app \
        --workers 4 \
        --worker-class uvicorn.workers.UvicornWorker \
        --bind 0.0.0.0:8000 \
        --timeout 300 \
        --access-logfile - \
        --error-logfile -
else
    echo "🛠️  Development modunda başlatılıyor..."
    echo "   Auto-reload: Aktif"
    echo "   Host: http://localhost:8000"
    echo ""
    echo "📚 API Dokümantasyonu:"
    echo "   Swagger UI: http://localhost:8000/docs"
    echo "   ReDoc: http://localhost:8000/redoc"
    echo ""
    echo "🧪 API'yi test etmek için:"
    echo "   python test_api.py"
    echo ""
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
fi
