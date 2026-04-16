from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Dict, List, Literal, Optional


class BaseSchema:
    def model_dump(self) -> Dict[str, Any]:
        return asdict(self)

    def model_dump_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.model_dump(), ensure_ascii=False, indent=indent)


class DatasetName(str, Enum):
    medquad = "medquad"
    medmcqa = "medmcqa"
    healthbench = "healthbench"


class TargetModel(str, Enum):
    gemini_31_pro_preview = "gemini-3.1"
    gpt_51 = "gpt-5.4"
    grok_41_fast_reasoning = "grok-4-1"
    mistral_large_3 = "mistral-large-latest"
    claude_sonnet_46 = "claude-sonnet-4-6"

    @property
    def display_name(self) -> str:
        return MODEL_DISPLAY_NAMES.get(self, self.value)

    @classmethod
    def parse(cls, value: "TargetModel | str") -> "TargetModel":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            raw = value.strip()
            if raw.startswith(f"{cls.__name__}."):
                raw = raw.split(".", 1)[1]
            # Root Cause vs Logic:
            # Root Cause: historical job payloads and docs used preview/legacy ids that no
            # longer match the canonical UI ids.
            # Logic: normalize those legacy ids into the current enum so old payloads stay valid
            # while all newly emitted payloads use the canonical compact names.
            legacy_aliases = {
                "gemini-3.1-pro-preview": cls.gemini_31_pro_preview,
                "gpt-5.1": cls.gpt_51,
                "grok-4-1-fast-reasoning": cls.grok_41_fast_reasoning,
                "sonnet-4.6": cls.claude_sonnet_46,
            }
            if raw in legacy_aliases:
                return legacy_aliases[raw]
            try:
                return cls(raw)
            except ValueError:
                if raw in cls.__members__:
                    return cls[raw]
        raise ValueError(f"Unknown target model: {value}")


MODEL_PROVIDER = {
    TargetModel.gemini_31_pro_preview: "google",
    TargetModel.gpt_51: "openai",
    TargetModel.grok_41_fast_reasoning: "xai",
    TargetModel.mistral_large_3: "mistral",
    TargetModel.claude_sonnet_46: "claude",
}

# Motivation vs Logic: keep UI and saved state stable by always presenting canonical vendor names.
MODEL_DISPLAY_NAMES = {
    TargetModel.gpt_51: "GPT 5.4",
    TargetModel.grok_41_fast_reasoning: "Grok 4.1",
    TargetModel.gemini_31_pro_preview: "Gemini-3.1",
    TargetModel.mistral_large_3: "Mistral Large",
    TargetModel.claude_sonnet_46: "Sonet 4.6",
}

@dataclass
class DatasetRow(BaseSchema):
    id: str
    prompt: List[Dict[str, str]]
    reference: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkRequest(BaseSchema):
    datasets: List[DatasetName]
    models: List[TargetModel]
    max_samples: int = 0
    workers: int = 10
    seed: int = 13
    output_subdir: Optional[str] = None
    enable_bert_score: bool = True

    def __post_init__(self) -> None:
        self.datasets = [item if isinstance(item, DatasetName) else DatasetName(item) for item in self.datasets]
        self.models = [TargetModel.parse(item) for item in self.models]


@dataclass
class JobSummary(BaseSchema):
    metric_means: Dict[str, float]
    rows: int
    artifacts: Dict[str, str]


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    paused = "paused"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


@dataclass
class JobInfo(BaseSchema):
    job_id: str
    status: JobStatus
    request: BenchmarkRequest
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    datasets: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class EventPayload(BaseSchema):
    event: str
    job_id: str
    ts: Optional[str] = None
    dataset: Optional[str] = None
    model: Optional[str] = None
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderResponse(BaseSchema):
    text: str
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LogRecord(BaseSchema):
    ts: str
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    job_id: str
    dataset: Optional[str] = None
    model: Optional[str] = None
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
