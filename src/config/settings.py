from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env', 
        env_file_encoding='utf-8',
        extra='ignore' 
    )
    DATABASE_URL:str
    GOOGLE_API_KEY:str
    AWS_ACCESS_KEY_ID:str
    AWS_SECRET_ACCESS_KEY:str
    AWS_REGION:str

@lru_cache
def get_settings() -> Settings:
    return Settings()

if __name__ == "__main__":
    settings = get_settings()
    print(settings.DATABASE_URL)
    print(settings.GOOGLE_API_KEY)
    print(settings.AWS_ACCESS_KEY_ID)
    print(settings.AWS_SECRET_ACCESS_KEY)
    print(settings.AWS_REGION)