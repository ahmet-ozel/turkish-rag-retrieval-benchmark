from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    """API ayarları"""
    
    # API Configuration
    api_title: str = "Doküman Chunking API"
    api_version: str = "1.0.0"
    debug: bool = False
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    
    # File Upload Limits
    max_upload_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: List[str] = ['txt', 'pdf', 'docx', 'csv', 'xlsx', 'xls']
    
    # Processing Limits
    max_chunk_size: int = 2000
    min_chunk_size: int = 100
    default_chunk_size: int = 500
    max_batch_files: int = 10
    
    # Redis Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    use_redis: bool = False
    
    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 60
    
    # CORS Settings
    cors_origins: List[str] = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["*"]
    cors_allow_headers: List[str] = ["*"]
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Monitoring
    enable_metrics: bool = True
    metrics_path: str = "/metrics"
    
    # Security
    secret_key: str = "your-secret-key-here-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    class Config:
        env_file = "config.env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Global settings instance
settings = Settings()

# Redis client (opsiyonel)
redis_client = None
if settings.use_redis:
    try:
        import redis
        redis_client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password if settings.redis_password else None,
            decode_responses=True
        )
        redis_client.ping()
        print("Redis bağlantısı başarılı")
    except Exception as e:
        print(f"Redis bağlantısı başarısız: {e}")
        redis_client = None
