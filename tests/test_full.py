import json
from pathlib import Path

from app.datasets import load_healthbench_rows, load_medmcqa_rows, load_medquad_rows
from app.metrics import compute_text_metrics, tok_f1
from app.models import BenchmarkRequest, TargetModel
from app.runner import parse_medmcqa_prediction


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')


def test_medquad_loader(tmp_path: Path):
    path = tmp_path / 'medquad.jsonl'
    write_jsonl(path, [{"id": "1", "question": "What is flu?", "answer": "A viral infection."}])
    rows = load_medquad_rows(path)
    assert rows[0].prompt[-1]['content'] == 'What is flu?'
    assert rows[0].reference == 'A viral infection.'


def test_medmcqa_loader_and_parser(tmp_path: Path):
    path = tmp_path / 'medmcqa.jsonl'
    write_jsonl(
        path,
        [{
            "id": "42",
            "question": "Which letter is correct?",
            "opa": "One",
            "opb": "Two",
            "opc": "Three",
            "opd": "Four",
            "cop": 2,
            "choice_type": "single",
        }],
    )
    rows = load_medmcqa_rows(path)
    assert rows[0].reference == 'b'
    letters, valid, source = parse_medmcqa_prediction('Final answer: b', rows[0].metadata['options'], 'single')
    assert letters == ['b']
    assert valid is True
    assert source == 'strict_letters'


def test_healthbench_loader(tmp_path: Path):
    path = tmp_path / 'healthbench.jsonl'
    write_jsonl(
        path,
        [{
            "prompt_id": "abc",
            "messages": [{"role": "user", "content": "How to treat fever?"}],
            "ideal_completions_data": {"ideal_completion": "Use antipyretics and hydration."},
        }],
    )
    rows = load_healthbench_rows(path)
    assert rows[0].id == 'abc'
    assert rows[0].reference == 'Use antipyretics and hydration.'


def test_metrics_are_non_negative():
    metrics = compute_text_metrics('viral infection', 'viral infection')
    assert metrics['rougeL_f'] == 1.0
    assert tok_f1('a b', 'a b') == 1.0
    assert all(value >= 0.0 for value in metrics.values())


def test_target_model_parse_accepts_legacy_enum_style_strings():
    assert TargetModel.parse('TargetModel.gpt_51') == TargetModel.gpt_51
    assert TargetModel.parse('gpt-5.1') == TargetModel.gpt_51
    assert TargetModel.parse('gemini-3.1-pro-preview') == TargetModel.gemini_31_pro_preview
    assert TargetModel.gpt_51.value == 'gpt-5.4'
    request = BenchmarkRequest(
        datasets=['medquad'],
        models=['TargetModel.gemini_31_pro_preview'],
    )
    assert request.models == [TargetModel.gemini_31_pro_preview]
