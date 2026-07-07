"""Centralised config loaded from .env via pydantic-settings.

WHY this file exists:
- One source of truth for paths, model names, seeds.
- Reading os.environ in every module spreads magic strings across the codebase.
- pydantic-settings gives us typed config + .env loading + validation in 10 lines.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Project root resolves to recruit-rag/
    project_root: Path = Path(__file__).resolve().parents[2]

    # Subdirectories (resolved against project_root in get_*_dir() helpers)
    data_dir: str = "data"
    models_dir: str = "models"
    indexes_dir: str = "indexes"

    # Embeddings
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # LLM — override via .env `OLLAMA_MODEL=llama3.1:8b` once the model is pulled.
    # Default is qwen2.5:7b because it ships in a smaller install and is
    # instruction-tuned with strong JSON adherence.
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"

    # Reproducibility
    seed: int = 42

    @property
    def data_path(self) -> Path:
        return self.project_root / self.data_dir

    @property
    def models_path(self) -> Path:
        return self.project_root / self.models_dir

    @property
    def indexes_path(self) -> Path:
        return self.project_root / self.indexes_dir


settings = Settings()
