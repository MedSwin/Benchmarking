from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from app.models import DatasetName, MODEL_PROVIDER, TargetModel

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv() -> None:
        return None

load_dotenv()


def _parse_bool_flag(env_name: str, default: bool) -> bool:
    raw = os.getenv(env_name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


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
    output_root: Path = Path(os.getenv("OUTPUT_ROOT", "output"))  # Motivation vs Logic: keep audits and benchmarks centralized under the shared output directory.

    openai_api_keys: str = os.getenv("OPENAI_API_KEYS", "")
    xai_api_keys: str = os.getenv("XAI_API_KEYS", "")
    google_api_keys: str = os.getenv("GOOGLE_API_KEYS", "")
    mistral_api_keys: str = os.getenv("MISTRAL_API_KEYS", "")
    openai_models: str = os.getenv("OPENAI_MODEL", "gpt-5.4")
    xai_models: str = os.getenv("XAI_MODEL", "grok-4-1")
    google_models: str = os.getenv("GOOGLE_MODEL", "gemini-3.1")
    mistral_models: str = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
    openai_enabled: bool = _parse_bool_flag("OPENAI", True)
    xai_enabled: bool = _parse_bool_flag("XAI", True)
    google_enabled: bool = _parse_bool_flag("GOOGLEAI", True)
    mistral_enabled: bool = _parse_bool_flag("MISTRALAI", True)

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

    medmcqa_enabled: bool = _parse_bool_flag("MEDMCQA", True)
    medquad_enabled: bool = _parse_bool_flag("MEDQUAD", True)
    healthbench_enabled: bool = _parse_bool_flag("HEALTHBENCH", True)

    bert_score_model_type: str = os.getenv("BERT_SCORE_MODEL_TYPE", "roberta-large")

    @staticmethod
    def _split_values(raw: str) -> List[str]:
        return [item.strip() for item in raw.split(",") if item.strip()]

    def _filter_enabled_provider_values(self, raw_values: Dict[str, str]) -> Dict[str, List[str]]:
        return {
            provider: self._split_values(raw_values[provider])
            for provider, enabled in self.enabled_providers.items()
            if enabled
        }

    @property
    def provider_keys(self) -> Dict[str, List[str]]:
        return self._filter_enabled_provider_values(
            {
                "openai": self.openai_api_keys,
                "xai": self.xai_api_keys,
                "google": self.google_api_keys,
                "mistral": self.mistral_api_keys,
            }
        )

    @property
    def provider_models(self) -> Dict[str, List[str]]:
        return self._filter_enabled_provider_values(
            {
                "openai": self.openai_models,
                "xai": self.xai_models,
                "google": self.google_models,
                "mistral": self.mistral_models,
            }
        )

    # Motivation vs Logic: env-controlled dataset toggles keep the UI, API, and runner aligned with deploy limits.
    # Motivation vs Logic: env-controlled provider flags keep runtime + UI aligned with available API credentials.
    @property
    def enabled_providers(self) -> Dict[str, bool]:
        return {
            "openai": self.openai_enabled,
            "xai": self.xai_enabled,
            "google": self.google_enabled,
            "mistral": self.mistral_enabled,
        }

    @property
    def enabled_models(self) -> List[Dict[str, str]]:
        models: List[Dict[str, str]] = []
        for target_model in TargetModel:
            provider = MODEL_PROVIDER[target_model]
            if not self.enabled_providers.get(provider):
                continue
            models.append(
                {
                    "id": target_model.value,
                    "provider": provider,
                    "display_name": target_model.display_name,
                }
            )
        return models

    @property
    def enabled_datasets(self) -> List[DatasetName]:
        available = [
            (DatasetName.medmcqa, self.medmcqa_enabled),
            (DatasetName.medquad, self.medquad_enabled),
            (DatasetName.healthbench, self.healthbench_enabled),
        ]
        return [dataset for dataset, enabled in available if enabled]

    def is_dataset_enabled(self, dataset: DatasetName) -> bool:
        return dataset in self.enabled_datasets


settings = Settings()
