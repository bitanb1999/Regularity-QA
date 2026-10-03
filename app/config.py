from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()  # also export .env to os.environ, for libraries that read it directly (HF_TOKEN)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_user: str
    postgres_password: str
    postgres_db: str
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str

    groq_api_key: str
    groq_base_url: str = "https://api.groq.com/openai/v1"
    extraction_model: str = "openai/gpt-oss-120b"
    router_model: str = "openai/gpt-oss-20b"  # Groq rate limits are per model; keeps 120b's budget for answers
    answer_model: str = "openai/gpt-oss-120b"

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Langfuse tracing is off unless both keys are set (see app/observability.py).
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
