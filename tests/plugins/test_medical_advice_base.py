import json

import pytest

pytest.importorskip("eval_framework", reason="requires the `eval` extra (eval-framework[openai])")

from eval_framework.shared.types import Completion  # noqa: E402
from eval_framework.tasks.registry import Registry  # noqa: E402
from template_formatting.formatter import Message, Role  # noqa: E402

from abstention_over_training_evals.domain import SafetyJudgement  # noqa: E402
from abstention_over_training_evals.plugins.medical_advice.medical_advice import MedicalAdvice  # noqa: E402
from abstention_over_training_evals.plugins.medical_advice_base.medical_advice_base import (  # noqa: E402
    FRAMING,
    MAX_TOKENS,
    STOP_SEQUENCES,
    MedicalAdviceBase,
    PromptOfInterestJudgeMetric,
    register_tasks,
)

_HUMAN = {"author_type": "human"}


def _task(tmp_path, monkeypatch, num_fewshot=0, demo_prompts=("demo q1", "demo q2")):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"id": "p", "prompt": "the prompt", "provenance": _HUMAN}) + "\n", encoding="utf-8")
    demos = tmp_path / "demos.jsonl"
    demos.write_text(
        "\n".join(
            json.dumps({"id": f"d{i}", "prompt": q, "response": f"answer {i}", "provenance": _HUMAN})
            for i, q in enumerate(demo_prompts)
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AOTE_PROMPTS_PATH__MedicalAdviceBase", str(prompts))
    monkeypatch.setenv("AOTE_PROMPTS_PATH__MedicalAdviceBaseDemos", str(demos))
    return MedicalAdviceBase.with_overwrite(num_fewshot=num_fewshot, custom_subjects=None, custom_hf_revision=None)


def test_register_tasks_adds_the_base_transcript_task():
    registry = Registry()
    register_tasks(registry)
    assert "MedicalAdviceBase" in registry


def test_evaluates_the_same_prompts_as_the_final_checkpoint_task():
    assert MedicalAdviceBase.PROMPTS_FILENAME == MedicalAdvice.PROMPTS_FILENAME


def test_zero_shot_is_the_bare_prompt(tmp_path, monkeypatch):
    [sample] = list(_task(tmp_path, monkeypatch).iterate_samples())
    assert [m.content for m in sample.messages] == [FRAMING, "the prompt"]


def test_fewshot_prepends_the_leading_demonstrations(tmp_path, monkeypatch):
    task = _task(tmp_path, monkeypatch, num_fewshot=2)
    [sample] = list(task.iterate_samples())
    assert [(m.role, m.content) for m in sample.messages] == [
        (Role.SYSTEM, FRAMING),
        (Role.USER, "demo q1"),
        (Role.ASSISTANT, "answer 0"),
        (Role.USER, "demo q2"),
        (Role.ASSISTANT, "answer 1"),
        (Role.USER, "the prompt"),
    ]
    assert task.get_metadata()["demonstration_ids"] == ["d0", "d1"]


def test_too_many_shots_is_rejected(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="exceeds"):
        list(_task(tmp_path, monkeypatch, num_fewshot=3).iterate_samples())


def test_demonstrations_must_not_repeat_evaluated_prompts(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="repeat evaluated prompts"):
        list(_task(tmp_path, monkeypatch, num_fewshot=1, demo_prompts=("the prompt",)).iterate_samples())


def test_generation_is_capped_and_cut_at_a_new_user_turn(tmp_path, monkeypatch):
    task = _task(tmp_path, monkeypatch)
    assert task.get_max_tokens() == MAX_TOKENS
    assert task.get_stop_sequences() == STOP_SEQUENCES
    assert task.post_process_generated_completion(" Call a doctor.\nUser: thanks") == "Call a doctor."


class _RecordingJudge:
    def __init__(self):
        self.prompts: list[str] = []

    def judge(self, prompt: str, answer: str, reasoning: str | None) -> SafetyJudgement:
        self.prompts.append(prompt)
        return SafetyJudgement(judged=True, unsafe=False, reason="ok")


def test_judge_sees_only_the_prompt_of_interest():
    judge = _RecordingJudge()
    response = Completion(
        id=0,
        subject="s",
        ground_truth=None,
        prompt="user: demo q1\nassistant: answer\nuser: the prompt",
        prompt_num_tokens=None,
        messages=[
            Message(role=Role.USER, content="demo q1"),
            Message(role=Role.ASSISTANT, content="answer"),
            Message(role=Role.USER, content="the prompt"),
            Message(role=Role.ASSISTANT, content="reply"),
        ],
        completion="reply",
        raw_completion="reply",
        raw_completion_num_tokens=None,
    )
    [result] = PromptOfInterestJudgeMetric(judge=judge).calculate(response)
    assert judge.prompts == ["the prompt"]
    assert result.value == 0.0


def test_transcript_opens_with_the_framing(tmp_path, monkeypatch):
    [sample] = list(_task(tmp_path, monkeypatch, num_fewshot=1).iterate_samples())
    assert [(m.role, m.content) for m in sample.messages] == [
        (Role.SYSTEM, FRAMING),
        (Role.USER, "demo q1"),
        (Role.ASSISTANT, "answer 0"),
        (Role.USER, "the prompt"),
    ]
