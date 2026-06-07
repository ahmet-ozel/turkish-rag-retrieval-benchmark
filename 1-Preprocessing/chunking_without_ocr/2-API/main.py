from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import asyncio
import aiofiles
import os
import json
import uuid
from datetime import datetime
import sys
import io

# Chunking modülünü import et
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chunking_api import ChunkingProcessor

app = FastAPI(
    title="Doküman Chunking API",
    description="Dokümanları anlamlı parçalara bölen API servisi",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# İşlem durumu için basit bir in-memory store (production'da Redis kullanın)
processing_status = {}

class ChunkingRequest(BaseModel):
    method: str = Field(
        default="Sabit Boyut",
        description="Chunking yöntemi",
        enum=["Sabit Boyut", "Ayırıcı Bazlı", "Cümle Bazlı", "Satır Bazlı"]
    )
    chunk_size: int = Field(default=500, ge=100, le=2000, description="Chunk boyutu (karakter)")
    chunk_overlap: int = Field(default=50, ge=0, le=200, description="Chunk örtüşmesi (karakter)")
    separator: str = Field(default="\n\n", description="Ayırıcı karakter")
    include_columns_in_text: bool = Field(default=False, description="CSV/Excel sütunlarını text'e ekle")

class ProcessingStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    message: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str

class ChunkResponse(BaseModel):
    chunk_id: int
    filename: str
    text: str
    length: int
    metadata: Dict[str, Any] = {}

class ProcessingResponse(BaseModel):
    job_id: str
    message: str
    status_url: str

# Chunking işlemci instance'ı
chunking_processor = ChunkingProcessor()

@app.get("/")
async def root():
    """API ana endpoint'i"""
    return {
        "message": "Doküman Chunking API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "process": "/api/v1/process",
            "status": "/api/v1/status/{job_id}",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Sağlık kontrolü"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/v1/process", response_model=ProcessingResponse)
async def process_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    method: str = Query(default="Sabit Boyut"),
    chunk_size: int = Query(default=500, ge=100, le=2000),
    chunk_overlap: int = Query(default=50, ge=0, le=200),
    separator: str = Query(default="\n\n"),
    include_columns_in_text: bool = Query(default=False)
):
    """
    Doküman yükle ve chunking işlemini başlat

    - **file**: Yüklenecek dosya (PDF, DOCX, TXT, CSV, XLSX)
    - **method**: Chunking yöntemi
    - **chunk_size**: Her chunk'ın maksimum karakter sayısı
    - **chunk_overlap**: Chunk'lar arası örtüşme
    - **separator**: Ayırıcı bazlı yöntem için kullanılacak karakter
    - **include_columns_in_text**: CSV/Excel için sütun adlarını text'e ekle
    """

    # Dosya uzantısını kontrol et
    allowed_extensions = ['txt', 'pdf', 'docx', 'csv', 'xlsx', 'xls']
    file_extension = file.filename.split('.')[-1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Desteklenmeyen dosya formatı: {file_extension}. Desteklenen formatlar: {', '.join(allowed_extensions)}"
        )

    # Benzersiz job ID oluştur
    job_id = str(uuid.uuid4())

    # İşlem durumunu başlat
    processing_status[job_id] = {
        "status": "processing",
        "progress": 0.0,
        "message": "İşlem başlatıldı",
        "result": None,
        "error": None,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }

    # Arka planda işlemi başlat
    background_tasks.add_task(
        process_file_async,
        job_id,
        file,
        ChunkingRequest(
            method=method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=separator,
            include_columns_in_text=include_columns_in_text
        )
    )

    return ProcessingResponse(
        job_id=job_id,
        message="Dosya işleme başlatıldı",
        status_url=f"/api/v1/status/{job_id}"
    )

async def process_file_async(job_id: str, file: UploadFile, request: ChunkingRequest):
    """Dosyayı arka planda işle"""
    try:
        # Dosyayı geçici olarak kaydet
        temp_file_path = f"/tmp/{job_id}_{file.filename}"

        async with aiofiles.open(temp_file_path, 'wb') as temp_file:
            content = await file.read()
            await temp_file.write(content)

        # İlerleme güncelle
        processing_status[job_id]["progress"] = 0.2
        processing_status[job_id]["message"] = "Dosya okunuyor..."
        processing_status[job_id]["updated_at"] = datetime.now().isoformat()

        # Chunking işlemini yap
        chunks, error = await asyncio.to_thread(
            chunking_processor.process_file,
            temp_file_path,
            file.filename,
            request.method,
            request.chunk_size,
            request.chunk_overlap,
            request.separator,
            request.include_columns_in_text
        )

        # Geçici dosyayı sil
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        if error:
            processing_status[job_id]["status"] = "error"
            processing_status[job_id]["error"] = error
            processing_status[job_id]["message"] = "İşlem başarısız"
        else:
            # Sonuçları hazırla
            result = {
                "total_chunks": len(chunks),
                "filename": file.filename,
                "chunks": chunks,
                "statistics": {
                    "total_characters": sum(len(chunk.get('text', '')) for chunk in chunks),
                    "average_chunk_size": sum(len(chunk.get('text', '')) for chunk in chunks) / len(chunks) if chunks else 0,
                    "method": request.method,
                    "chunk_size": request.chunk_size,
                    "chunk_overlap": request.chunk_overlap
                }
            }

            processing_status[job_id]["status"] = "completed"
            processing_status[job_id]["progress"] = 1.0
            processing_status[job_id]["message"] = "İşlem tamamlandı"
            processing_status[job_id]["result"] = result

        processing_status[job_id]["updated_at"] = datetime.now().isoformat()

    except Exception as e:
        processing_status[job_id]["status"] = "error"
        processing_status[job_id]["error"] = str(e)
        processing_status[job_id]["message"] = "Beklenmeyen hata oluştu"
        processing_status[job_id]["updated_at"] = datetime.now().isoformat()

@app.get("/api/v1/status/{job_id}", response_model=ProcessingStatus)
async def get_processing_status(job_id: str):
    """İşlem durumunu sorgula"""

    if job_id not in processing_status:
        raise HTTPException(
            status_code=404,
            detail="İşlem bulunamadı"
        )

    status = processing_status[job_id]

    return ProcessingStatus(
        job_id=job_id,
        status=status["status"],
        progress=status["progress"],
        message=status["message"],
        result=status["result"],
        error=status["error"],
        created_at=status["created_at"],
        updated_at=status["updated_at"]
    )

@app.post("/api/v1/process-sync")
async def process_document_sync(
    file: UploadFile = File(...),
    method: str = Query(default="Sabit Boyut"),
    chunk_size: int = Query(default=500, ge=100, le=2000),
    chunk_overlap: int = Query(default=50, ge=0, le=200),
    separator: str = Query(default="\n\n"),
    include_columns_in_text: bool = Query(default=False)
):
    """
    Dokümanı senkron olarak işle (küçük dosyalar için)

     Büyük dosyalar için /api/v1/process endpoint'ini kullanın
    """

    # Dosya uzantısını kontrol et
    allowed_extensions = ['txt', 'pdf', 'docx', 'csv', 'xlsx', 'xls']
    file_extension = file.filename.split('.')[-1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Desteklenmeyen dosya formatı: {file_extension}"
        )

    # Dosya boyutunu kontrol et (5MB limit)
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size > 5 * 1024 * 1024:  # 5MB
        raise HTTPException(
            status_code=413,
            detail="Dosya çok büyük. Lütfen async endpoint kullanın: /api/v1/process"
        )

    try:
        # Geçici dosya oluştur
        temp_file_path = f"/tmp/sync_{uuid.uuid4()}_{file.filename}"

        content = await file.read()
        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(content)

        # Chunking işlemi
        chunks, error = await asyncio.to_thread(
            chunking_processor.process_file,
            temp_file_path,
            file.filename,
            method,
            chunk_size,
            chunk_overlap,
            separator,
            include_columns_in_text
        )

        # Geçici dosyayı sil
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        if error:
            raise HTTPException(status_code=400, detail=error)

        return {
            "success": True,
            "filename": file.filename,
            "total_chunks": len(chunks),
            "chunks": chunks,
            "statistics": {
                "total_characters": sum(len(chunk.get('text', '')) for chunk in chunks),
                "average_chunk_size": sum(len(chunk.get('text', '')) for chunk in chunks) / len(chunks) if chunks else 0,
                "method": method
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"İşlem sırasında hata oluştu: {str(e)}"
        )

@app.post("/api/v1/batch-process")
async def batch_process_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    method: str = Query(default="Sabit Boyut"),
    chunk_size: int = Query(default=500, ge=100, le=2000),
    chunk_overlap: int = Query(default=50, ge=0, le=200),
    separator: str = Query(default="\n\n"),
    include_columns_in_text: bool = Query(default=False)
):
    """
    Birden fazla dokümanı toplu olarak işle

    Maksimum 10 dosya aynı anda işlenebilir.
    """

    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maksimum 10 dosya aynı anda işlenebilir"
        )

    # Her dosya için job oluştur
    jobs = []

    for file in files:
        # Dosya uzantısını kontrol et
        file_extension = file.filename.split('.')[-1].lower()
        allowed_extensions = ['txt', 'pdf', 'docx', 'csv', 'xlsx', 'xls']

        if file_extension not in allowed_extensions:
            jobs.append({
                "filename": file.filename,
                "job_id": None,
                "error": f"Desteklenmeyen format: {file_extension}"
            })
            continue

        # Job ID oluştur
        job_id = str(uuid.uuid4())

        # İşlem durumunu başlat
        processing_status[job_id] = {
            "status": "processing",
            "progress": 0.0,
            "message": "İşlem başlatıldı",
            "result": None,
            "error": None,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }

        # Arka plan görevini ekle
        background_tasks.add_task(
            process_file_async,
            job_id,
            file,
            ChunkingRequest(
                method=method,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator=separator,
                include_columns_in_text=include_columns_in_text
            )
        )

        jobs.append({
            "filename": file.filename,
            "job_id": job_id,
            "status_url": f"/api/v1/status/{job_id}"
        })

    return {
        "message": f"{len(files)} dosya işleme alındı",
        "jobs": jobs
    }

@app.get("/api/v1/export/{job_id}")
async def export_chunks(
    job_id: str,
    format: str = Query(default="json", enum=["json", "txt", "csv"])
):
    """İşlem sonuçlarını farklı formatlarda dışa aktar"""

    if job_id not in processing_status:
        raise HTTPException(status_code=404, detail="İşlem bulunamadı")

    status = processing_status[job_id]

    if status["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"İşlem henüz tamamlanmadı. Durum: {status['status']}"
        )

    result = status["result"]
    chunks = result["chunks"]

    if format == "json":
        return JSONResponse(
            content=result,
            headers={
                "Content-Disposition": f"attachment; filename=chunks_{job_id}.json"
            }
        )

    elif format == "txt":
        output = []
        for i, chunk in enumerate(chunks, 1):
            output.append(f"=== CHUNK {i} ===")
            output.append(f"Dosya: {chunk.get('filename', 'unknown')}")
            output.append(f"Uzunluk: {len(chunk.get('text', ''))}")
            output.append("İçerik:")
            output.append(chunk.get('text', ''))
            output.append("\n" + "=" * 50 + "\n")

        content = "\n".join(output)

        return StreamingResponse(
            io.StringIO(content),
            media_type="text/plain",
            headers={
                "Content-Disposition": f"attachment; filename=chunks_{job_id}.txt"
            }
        )

    elif format == "csv":
        import csv
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        headers = ['chunk_id', 'filename', 'text', 'length']
        if chunks and isinstance(chunks[0], dict):
            # Ek sütunları bul
            extra_columns = [k for k in chunks[0].keys() if k not in ['text', 'filename']]
            headers.extend(extra_columns)

        writer.writerow(headers)

        # Data
        for i, chunk in enumerate(chunks, 1):
            row = [
                i,
                chunk.get('filename', ''),
                chunk.get('text', ''),
                len(chunk.get('text', ''))
            ]

            # Ek sütunları ekle
            if 'extra_columns' in locals():
                for col in extra_columns:
                    row.append(chunk.get(col, ''))

            writer.writerow(row)

        output.seek(0)

        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=chunks_{job_id}.csv"
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
