from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.metrics import norm_text
from app.models import DatasetName, DatasetRow

INSTR = (
    "You are a careful clinical assistant. Answer the patient question using "
    "general, authoritative medical knowledge. Be concise (<=150 words), avoid speculation. "
    "If unsure, say: I don't know."
)

SYS_MSG_MEDMCQA = (
    "You are a careful medical exam assistant. "
    "You must answer using only option letter(s). "
    "Never explain your reasoning. Never write full words unless they are already encoded as option letters."
)

LETTER_BY_INDEX = {1: "a", 2: "b", 3: "c", 4: "d"}
VALID_LETTERS = {"a", "b", "c", "d"}


def _guard_lfs_pointer(path: Path) -> None:
    with path.open("r", encoding="utf-8") as handle:
        prefix = handle.readline().strip()
    if prefix == "version https://git-lfs.github.com/spec/v1":
        raise RuntimeError(
            f"{path} is a Git LFS pointer, not the dataset payload. Run `git lfs pull` before benchmarking."
        )


def dataset_path(root: Path, dataset: DatasetName) -> Path:
    return root / dataset.value / f"{dataset.value}.jsonl"


def _sample_rows(rows: List[DatasetRow], max_samples: int, seed: int) -> List[DatasetRow]:
    random.seed(seed)
    random.shuffle(rows)
    if max_samples and max_samples > 0:
        return rows[:max_samples]
    return rows


def load_dataset_rows(root: Path, dataset: DatasetName, max_samples: int, seed: int) -> List[DatasetRow]:
    path = dataset_path(root, dataset)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    _guard_lfs_pointer(path)
    loader = {
        DatasetName.medquad: load_medquad_rows,
        DatasetName.medmcqa: load_medmcqa_rows,
        DatasetName.healthbench: load_healthbench_rows,
    }[dataset]
    return _sample_rows(loader(path), max_samples, seed)


def load_medquad_rows(path: Path) -> List[DatasetRow]:
    rows: List[DatasetRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            obj = json.loads(line)
            question = norm_text(obj.get("question"))
            answer = norm_text(obj.get("answer"))
            if not question or not answer:
                continue
            rows.append(
                DatasetRow(
                    id=str(obj.get("id") or f"row-{line_no}"),
                    prompt=[
                        {"role": "system", "content": INSTR},
                        {"role": "user", "content": question},
                    ],
                    reference=answer,
                    metadata={
                        "question": question,
                    },
                )
            )
    return rows


def normalize_choice_type(value: Any) -> str:
    return "multi" if norm_text(value).lower() == "multi" else "single"


def uniq_sorted_letters(items: Iterable[str]) -> List[str]:
    return sorted({item.lower() for item in items if item and item.lower() in VALID_LETTERS})


def normalize_option_text(value: Any) -> str:
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", norm_text(value).lower()).split())


def cop_piece_to_letters(piece: Any, option_map: Dict[str, str]) -> List[str]:
    if piece is None:
        return []
    if isinstance(piece, int):
        return [LETTER_BY_INDEX[piece]] if piece in LETTER_BY_INDEX else []
    if isinstance(piece, float) and piece.is_integer():
        digit = int(piece)
        return [LETTER_BY_INDEX[digit]] if digit in LETTER_BY_INDEX else []
    if isinstance(piece, list):
        letters: List[str] = []
        for item in piece:
            letters.extend(cop_piece_to_letters(item, option_map))
        return uniq_sorted_letters(letters)

    text = norm_text(piece)
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, (list, int, float, str)):
            return cop_piece_to_letters(parsed, option_map)
    except Exception:
        pass

    letters: List[str] = []
    slow = text.lower()
    for digit in re.findall(r"\b([1-4])\b", slow):
        letters.append(LETTER_BY_INDEX[int(digit)])
    letters.extend(re.findall(r"\b([abcd])\b", slow))

    normalized = normalize_option_text(text)
    for letter, option_text in option_map.items():
        if option_text and normalized == normalize_option_text(option_text):
            letters.append(letter)
    return uniq_sorted_letters(letters)


def parse_gold_answers(obj: Dict[str, Any], option_map: Dict[str, str]) -> List[str]:
    gold = cop_piece_to_letters(obj.get("cop"), option_map)
    if gold:
        return gold
    for key in ["answer", "answers", "correct_option", "correct_options"]:
        gold = cop_piece_to_letters(obj.get(key), option_map)
        if gold:
            return gold
    return []


def build_medmcqa_prompt(question: str, option_map: Dict[str, str], choice_type: str) -> str:
    if choice_type == "single":
        task = (
            "Task: choose the single best answer.\n"
            "Output format: exactly one lowercase letter: a or b or c or d.\n"
            "Return only the letter."
        )
    else:
        task = (
            "Task: choose all correct answers.\n"
            "Output format: lowercase letters only, sorted alphabetically, separated by commas.\n"
            "Valid examples: a,c   b,d   a,b,c\n"
            "Return only the letters."
        )
    return (
        f"Question: {question}\n\n"
        f"Options:\n"
        f"a. {option_map['a']}\n"
        f"b. {option_map['b']}\n"
        f"c. {option_map['c']}\n"
        f"d. {option_map['d']}\n\n"
        f"{task}\n\n"
        f"Answer:"
    )


def load_medmcqa_rows(path: Path) -> List[DatasetRow]:
    rows: List[DatasetRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            obj = json.loads(line)
            option_map = {letter: norm_text(obj.get(f"op{letter}")) for letter in ["a", "b", "c", "d"]}
            choice_type = normalize_choice_type(obj.get("choice_type"))
            gold_letters = parse_gold_answers(obj, option_map)
            question = norm_text(obj.get("question"))
            if not question or not all(option_map.values()) or not gold_letters:
                continue
            prompt = build_medmcqa_prompt(question, option_map, choice_type)
            rows.append(
                DatasetRow(
                    id=str(obj.get("id") or f"row-{line_no}"),
                    prompt=[
                        {"role": "system", "content": SYS_MSG_MEDMCQA},
                        {"role": "user", "content": prompt},
                    ],
                    reference=",".join(gold_letters),
                    metadata={
                        "choice_type": choice_type,
                        "question": question,
                        "options": option_map,
                        "gold_letters": gold_letters,
                        "gold_texts": [option_map[item] for item in gold_letters],
                        "subject_name": norm_text(obj.get("subject_name")),
                        "topic_name": norm_text(obj.get("topic_name")),
                    },
                )
            )
    return rows


def _content_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join([piece for piece in (_content_to_text(item) for item in value) if norm_text(piece)])
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return value["text"]
        if "content" in value:
            return _content_to_text(value["content"])
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def extract_reference(obj: Dict[str, Any]) -> Optional[str]:
    ideal = obj.get("ideal_completions_data") or {}
    primary = norm_text(_content_to_text(ideal.get("ideal_completion")))
    if primary:
        return primary
    for item in ideal.get("ideal_completions_ref_completions") or []:
        candidate = norm_text(_content_to_text(item))
        if candidate:
            return candidate
    for key in [
        "processed_ideal_completion_en_plaintext",
        "ideal_completion",
        "answer",
        "reference",
        "gold",
        "gold_answer",
    ]:
        candidate = norm_text(_content_to_text(obj.get(key)))
        if candidate:
            return candidate
    return None


def extract_messages(obj: Dict[str, Any]) -> List[Dict[str, str]]:
    candidates = [obj.get("prompt"), obj.get("messages"), obj.get("processed_prompt_en_plaintext"), obj.get("prompt_text")]
    for raw in candidates:
        cleaned: List[Dict[str, str]] = []
        if raw is None:
            continue
        if isinstance(raw, str):
            text = norm_text(raw)
            if text:
                return [{"role": "user", "content": text}]
            continue
        if isinstance(raw, dict):
            role = str(raw.get("role", "user")).strip().lower() or "user"
            content = norm_text(_content_to_text(raw.get("content", raw)))
            if content:
                return [{"role": role, "content": content}]
            continue
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    role = str(item.get("role", "user")).strip().lower() or "user"
                    content = norm_text(_content_to_text(item.get("content", "")))
                    if content:
                        cleaned.append({"role": role, "content": content})
                else:
                    content = norm_text(_content_to_text(item))
                    if content:
                        cleaned.append({"role": "user", "content": content})
            if cleaned:
                return cleaned
    return []


def load_healthbench_rows(path: Path) -> List[DatasetRow]:
    rows: List[DatasetRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            obj = json.loads(line)
            reference = extract_reference(obj)
            messages = extract_messages(obj)
            if not reference or not messages:
                continue
            rows.append(
                DatasetRow(
                    id=str(obj.get("prompt_id") or obj.get("id") or f"row-{line_no}"),
                    prompt=[{"role": "system", "content": INSTR}, *messages],
                    reference=reference,
                    metadata={
                        "rubrics": obj.get("rubrics", []),
                        "example_tags": obj.get("example_tags", []),
                    },
                )
            )
    return rows
