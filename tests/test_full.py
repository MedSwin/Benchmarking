import json
import asyncio
from pathlib import Path

from app.datasets import load_healthbench_rows, load_medmcqa_rows, load_medquad_rows
from app.metrics import compute_text_metrics, tok_f1
from app.models import BenchmarkRequest, DatasetName, DatasetRow, ProviderResponse, TargetModel
from app.runner import BenchmarkManager, parse_medmcqa_prediction


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
    assert TargetModel.parse('sonnet-4.6') == TargetModel.claude_sonnet_46
    assert TargetModel.gpt_51.value == 'gpt-5.4'
    assert TargetModel.claude_sonnet_46.display_name == 'Sonet 4.6'
    request = BenchmarkRequest(
        datasets=['medquad'],
        models=['TargetModel.gemini_31_pro_preview'],
    )
    assert request.models == [TargetModel.gemini_31_pro_preview]


def test_row_scored_event_uses_finalized_bert_score(tmp_path: Path, monkeypatch):
    manager = BenchmarkManager()
    job_id = "job-test-bert"
    manager.event_queues[job_id] = asyncio.Queue()

    class _FakeTensor:
        def __init__(self, values):
            self._values = values

        def tolist(self):
            return self._values

    def fake_bert_score(*_args, **_kwargs):
        return None, None, _FakeTensor([0.77])

    async def fake_generate(_model, _messages, _max_tokens):
        return ProviderResponse(text="A viral infection.")

    monkeypatch.setattr("app.runner._get_bert_score_fn", lambda: fake_bert_score)
    monkeypatch.setattr(manager.provider_pool, "generate", fake_generate)

    async def run_eval():
        rows = [
            DatasetRow(
                id="row-1",
                prompt=[{"role": "user", "content": "What is flu?"}],
                reference="A viral infection.",
            )
        ]
        final_rows, _summary, _model_error = await manager._evaluate_model(
            job_id=job_id,
            dataset=DatasetName.medquad,
            model=TargetModel.gpt_51,
            rows=rows,
            dataset_dir=tmp_path,
            workers=1,
            enable_bert_score=True,
        )
        return final_rows

    final_rows = asyncio.run(run_eval())
    assert final_rows[0]["bert_f"] == 0.77

    events = []
    queue = manager.event_queues[job_id]
    while not queue.empty():
        events.append(queue.get_nowait())

    row_generated = [event for event in events if event.event == "row_generated"]
    row_scored = [event for event in events if event.event == "row_scored"]
    assert len(row_generated) == 1
    assert len(row_scored) == 1
    assert row_scored[0].data["row_id"] == "row-1"
    assert row_scored[0].data["bert_f"] == 0.77

    asyncio.run(manager.shutdown())
