from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv() -> None:
        return None

load_dotenv()


@dataclass
class Settings:
    app_name: str = os.getenv("APP_NAME", "Medical Benchmark Control Plane")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    default_workers: int = int(os.getenv("DEFAULT_WORKERS", "10"))
    event_queue_size: int = int(os.getenv("EVENT_QUEUE_SIZE", "2000"))
    request_timeout_seconds: float = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "180"))
    retry_attempts: int = int(os.getenv("RETRY_ATTEMPTS", "3"))
    retry_base_delay_seconds: float = float(os.getenv("RETRY_BASE_DELAY_SECONDS", "2"))

    data_root: Path = Path(os.getenv("DATA_ROOT", "data"))
    output_root: Path = Path(os.getenv("OUTPUT_ROOT", "runs"))

    openai_api_keys: str = os.getenv("OPENAI_API_KEYS", "")
    xai_api_keys: str = os.getenv("XAI_API_KEYS", "")
    google_api_keys: str = os.getenv("GOOGLE_API_KEYS", "")
    mistral_api_keys: str = os.getenv("MISTRAL_API_KEYS", "")

    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    xai_base_url: str = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
    google_base_url: str = os.getenv("GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
    mistral_base_url: str = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1")

    openai_requests_per_minute: int = int(os.getenv("OPENAI_REQUESTS_PER_MINUTE", "60"))
    xai_requests_per_minute: int = int(os.getenv("XAI_REQUESTS_PER_MINUTE", "60"))
    google_requests_per_minute: int = int(os.getenv("GOOGLE_REQUESTS_PER_MINUTE", "60"))
    mistral_requests_per_minute: int = int(os.getenv("MISTRAL_REQUESTS_PER_MINUTE", "60"))

    medmcqa_max_new_tokens: int = int(os.getenv("MEDMCQA_MAX_NEW_TOKENS", "8"))
    medquad_max_new_tokens: int = int(os.getenv("MEDQUAD_MAX_NEW_TOKENS", "256"))
    healthbench_max_new_tokens: int = int(os.getenv("HEALTHBENCH_MAX_NEW_TOKENS", "256"))

    bert_score_model_type: str = os.getenv("BERT_SCORE_MODEL_TYPE", "roberta-large")

    @staticmethod
    def _split_keys(raw: str) -> List[str]:
        return [item.strip() for item in raw.split(",") if item.strip()]

    @property
    def provider_keys(self) -> Dict[str, List[str]]:
        return {
            "openai": self._split_keys(self.openai_api_keys),
            "xai": self._split_keys(self.xai_api_keys),
            "google": self._split_keys(self.google_api_keys),
            "mistral": self._split_keys(self.mistral_api_keys),
        }


settings = Settings()
