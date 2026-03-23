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
    gemini_31_pro_preview = "gemini-3.1-pro-preview"
    gpt_51 = "gpt-5.1"
    grok_41_fast_reasoning = "grok-4-1-fast-reasoning"
    mistral_large_3 = "mistral-large-latest"


MODEL_PROVIDER = {
    TargetModel.gemini_31_pro_preview: "google",
    TargetModel.gpt_51: "openai",
    TargetModel.grok_41_fast_reasoning: "xai",
    TargetModel.mistral_large_3: "mistral",
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
        self.models = [item if isinstance(item, TargetModel) else TargetModel(item) for item in self.models]


@dataclass
class JobSummary(BaseSchema):
    metric_means: Dict[str, float]
    rows: int
    artifacts: Dict[str, str]


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
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
