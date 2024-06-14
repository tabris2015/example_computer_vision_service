from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import cache

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    
    api_name: str = "Selfie processing service"
    revision: str = "local"
    yolo_version: str = "yolov8x-seg.pt"
    blur_filter_factor: float = 0.031
    log_level: str = "DEBUG"


@cache
def get_settings():
    print("getting settings...")
    return Settings()