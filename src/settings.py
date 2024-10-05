from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Discord bot token
    discord_token: str

    # LLMs for generating text
    openai_api_key: str

    # Storing embeddings
    pinecone_api_key: str

    class Config:
        """
        Used by the BaseSettings superclass as config options
        """
        env_file = ".env"
        case_sensitive = False


SETTINGS = Settings()
