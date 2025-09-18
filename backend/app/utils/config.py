"""
Configuration management for the Perplexity AI clone
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    app_name: str = "Perplexity AI Clone"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    
    # Database Configuration
    database_url: str = Field(default="sqlite:///./perplexity_clone.db", env="DATABASE_URL")
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # AI/LLM Configuration
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    groq_api_key: Optional[str] = Field(default=None, env="GROQ_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    default_llm_model: str = Field(default="llama-3.3-70b-versatile", env="DEFAULT_LLM_MODEL")
    
    # Vector Database Configuration
    chroma_persist_directory: str = Field(default="./data/chroma", env="CHROMA_PERSIST_DIRECTORY")
    pinecone_api_key: Optional[str] = Field(default=None, env="PINECONE_API_KEY")
    pinecone_environment: Optional[str] = Field(default=None, env="PINECONE_ENVIRONMENT")
    
    # Search Configuration
    elasticsearch_url: str = Field(default="http://localhost:9200", env="ELASTICSEARCH_URL")
    max_search_results: int = Field(default=20, env="MAX_SEARCH_RESULTS")
    
    # Web Scraping Configuration
    user_agent: str = Field(
        default="Perplexity-Clone-Bot/1.0",
        env="USER_AGENT"
    )
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")
    max_concurrent_requests: int = Field(default=10, env="MAX_CONCURRENT_REQUESTS")
    
    # Processing Configuration
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    max_file_size: int = Field(default=50_000_000, env="MAX_FILE_SIZE")  # 50MB
    
    # Cache Configuration
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")  # 1 hour
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=3600, env="RATE_LIMIT_WINDOW")  # 1 hour
    
    # CORS Configuration
    allowed_origins: list = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        env="ALLOWED_ORIGINS"
    )
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    
    # Security
    secret_key: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global settings instance
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """Get application settings (singleton pattern)"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

def reload_settings():
    """Reload settings (useful for testing)"""
    global _settings
    _settings = None
    return get_settings()

# Environment detection
def is_development() -> bool:
    """Check if running in development mode"""
    return get_settings().debug

def is_production() -> bool:
    """Check if running in production mode"""
    return not is_development()

# Configuration validation
def validate_config():
    """Validate critical configuration settings"""
    settings = get_settings()
    
    errors = []
    
    # Check required API keys
    if not settings.openai_api_key and not settings.anthropic_api_key:
        errors.append("At least one LLM API key (OpenAI or Anthropic) must be provided")
    
    # Check database URL format
    if not settings.database_url.startswith(('sqlite://', 'postgresql://', 'mysql://')):
        errors.append("Invalid database URL format")
    
    # Check directories exist or can be created
    try:
        os.makedirs(settings.chroma_persist_directory, exist_ok=True)
    except Exception as e:
        errors.append(f"Cannot create Chroma directory: {e}")
    
    if errors:
        raise ValueError(f"Configuration errors: {'; '.join(errors)}")
    
    return True