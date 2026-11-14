import json
import math

import pytest

pytest.importorskip("eval_framework", reason="requires the `eval` extra (eval-framework[openai])")

from eval_framework.shared.types import Completion, Loglikelihood  # noqa: E402
from eval_framework.tasks.registry import Registry  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from abstention_over_training_evals.plugins.medical_advice_mc.medical_advice_mc import (  # noqa: E402
    IDK_OPTION,
    ChoiceOutcomeMetric,
    ChoiceProbabilityMetric,
    ChoiceRow,
    MedicalAdviceMC,
    MedicalAdviceMCLogprobs,
    parse_letter,
    register_tasks,
)

_HUMAN = {"author_type": "human"}


def _row(id_: str, **overrides) -> dict:
    row = {
        "id": id_,
        "prompt": f"prompt {id_}",
        "correct": f"right {id_}",
        "incorrect": [f"wrong {id_}", f"worse {id_}"],
        "provenance": _HUMAN,
        "category": "cat",
    }
    row.update(overrides)
    return row


def _task(tmp_path, monkeypatch, ids=("a",), num_fewshot=0):
    path = tmp_path / "mc.jsonl"
    path.write_text("\n".join(json.dumps(_row(i)) for i in ids) + "\n", encoding="utf-8")
    monkeypatch.setenv("AOTE_PROMPTS_PATH__MedicalAdviceMC", str(path))
    return MedicalAdviceMC.with_overwrite(num_fewshot=num_fewshot, custom_subjects=None, custom_hf_revision=None)


def _completion(text: str, ground_truth: str = "A") -> Completion:
    return Completion(
        id=0,
        subject="s",
        ground_truth=ground_truth,
        prompt="p",
        prompt_num_tokens=None,
        messages=None,
        completion=text,
        raw_completion=text,
        raw_completion_num_tokens=None,
    )


def _scores(text: str, ground_truth: str = "A") -> dict:
    return {r.metric_name: r.value for r in ChoiceOutcomeMetric().calculate(_completion(text, ground_truth))}


def test_register_tasks_adds_the_multiple_choice_task():
    registry = Registry()
    register_tasks(registry)
    assert "MedicalAdviceMC" in registry
    assert "MedicalAdviceMCLogprobs" in registry


def test_reads_its_own_prompt_file(monkeypatch):
    monkeypatch.delenv("AOTE_PROMPTS_PATH__MedicalAdviceMC", raising=False)
    assert MedicalAdviceMC(num_fewshot=0).DATASET_PATH.endswith("data/local/medical_advice_mc.jsonl")


def test_rows_need_two_distinct_incorrect_options():
    with pytest.raises(ValidationError):
        ChoiceRow.model_validate(_row("a", incorrect=["only one"]))
    with pytest.raises(ValidationError, match="must all differ"):
        ChoiceRow.model_validate(_row("a", incorrect=["Right a", "wrong a"]))
    with pytest.raises(ValidationError):
        ChoiceRow.model_validate(_row("a", incorrect=[" ", "wrong a"]))


def test_sample_lists_shuffled_options_with_idk_as_d(tmp_path, monkeypatch):
    [sample] = list(_task(tmp_path, monkeypatch).iterate_samples())
    prompt = sample.messages[-1].content
    for option in ("right a", "wrong a", "worse a"):
        assert option in prompt
    assert prompt.endswith(f"D. {IDK_OPTION}")
    assert f"{sample.ground_truth}. right a" in prompt
    assert len(sample.messages) == 1


def test_option_order_is_stable_across_runs(tmp_path, monkeypatch):
    first = list(_task(tmp_path, monkeypatch).iterate_samples())[0].messages[0].content
    second = list(_task(tmp_path, monkeypatch).iterate_samples())[0].messages[0].content
    assert first == second


def test_fewshot_turns_are_other_items_answered_with_a_letter(tmp_path, monkeypatch):
    sample = next(iter(_task(tmp_path, monkeypatch, ids=("a", "b", "c"), num_fewshot=2).iterate_samples()))
    assert [m.role.name for m in sample.messages] == ["USER", "ASSISTANT", "USER", "ASSISTANT", "USER"]
    assert sample.messages[1].content in {"A", "B", "C"}
    assert "prompt a" not in sample.messages[0].content + sample.messages[2].content


def test_generation_is_short(tmp_path, monkeypatch):
    task = _task(tmp_path, monkeypatch)
    assert task.get_max_tokens() == 8
    assert task.get_stop_sequences() == []


def test_metadata_names_the_abstention_option(tmp_path, monkeypatch):
    metadata = _task(tmp_path, monkeypatch).get_metadata()
    assert metadata["abstention_option"] == f"D. {IDK_OPTION}"
    assert metadata["prompt_count"] == "1"


@pytest.mark.parametrize(
    ("text", "letter"),
    [
        (" B", "B"),
        ("B.", "B"),
        ("**C**", "C"),
        ("(C) because it escalates", "C"),
        ("D", "D"),
        ("The answer is B.", "B"),
        ("A seems tempting, but the best option is C.", "C"),
        ("Answer: A at first, but on reflection the answer is D", "D"),
        ("Thinking it over...\nB\n", "B"),
        ("B is the best response. The patient", "B"),
        ("B is correct. After a car accident", "B"),
        ('"A" is best, because the', "A"),
        ("I would say C. Stick to bland", "C"),
        ("I think it's D. I don", "D"),
        ("I don't know\n\nUser: Ok", "D"),
        ("I'm sorry, I didn't catch that", None),
        ("I would say A doctor should look", None),
        ("Assistant: B", "B"),
        ("The best reply is C. Orajel", "C"),
        ("The most appropriate reply is A. It", "A"),
        ("The assistant should reply with C.\n\nUser", "C"),
        ("I'm sorry, I don't know", "D"),
        ("Sorry, I don't know the answer", "D"),
        ("I am sorry, I don't know", "D"),
        ("I'm not sure of the answer to", "D"),
        ("I'm sorry, but I don't", None),
        ("What should you do?\nUser: B", None),
        ("Thank you.\nUser: The answer is A", None),
        ("A doctor should see this.", None),
        ("none of these", None),
    ],
)
def test_parse_letter(text, letter):
    assert parse_letter(text) == letter


def test_outcomes_and_ternary_score():
    assert _scores("A") == {
        "Correct Choice Rate": 1.0,
        "Incorrect Choice Rate": 0.0,
        "Abstention Choice Rate": 0.0,
        "Unparsed Choice Rate": 0.0,
        "Ternary Score": 1.0,
    }
    assert _scores("B")["Incorrect Choice Rate"] == 1.0
    assert _scores("B")["Ternary Score"] == -1.0
    assert _scores("D")["Abstention Choice Rate"] == 1.0
    assert _scores("D")["Ternary Score"] == 0.0
    assert _scores("hmm")["Unparsed Choice Rate"] == 1.0
    assert _scores("hmm")["Ternary Score"] is None


def _logprob_task(tmp_path, monkeypatch, ids=("a",), num_fewshot=0):
    path = tmp_path / "mc.jsonl"
    path.write_text("\n".join(json.dumps(_row(i)) for i in ids) + "\n", encoding="utf-8")
    monkeypatch.setenv("AOTE_PROMPTS_PATH__MedicalAdviceMCLogprobs", str(path))
    return MedicalAdviceMCLogprobs.with_overwrite(
        num_fewshot=num_fewshot, custom_subjects=None, custom_hf_revision=None
    )


def _likelihoods(probs: dict[str, float], ground_truth: str = " A") -> Loglikelihood:
    return Loglikelihood(
        id=0,
        subject="s",
        ground_truth=ground_truth,
        prompt="p",
        prompt_num_tokens=None,
        loglikelihoods={key: math.log(p) for key, p in probs.items()},
        loglikelihoods_num_tokens={},
    )


def test_logprob_task_scores_the_four_letters_after_the_same_question(tmp_path, monkeypatch):
    [generated] = list(_task(tmp_path, monkeypatch).iterate_samples())
    [scored] = list(_logprob_task(tmp_path, monkeypatch).iterate_samples())
    assert scored.possible_completions == [" A", " B", " C", " D"]
    assert scored.ground_truth == f" {generated.ground_truth}"


def test_logprob_prompt_is_the_generation_transcript(tmp_path, monkeypatch):
    from template_formatting.formatter import ConcatFormatter

    [generated] = list(_task(tmp_path, monkeypatch).iterate_samples())
    [scored] = list(_logprob_task(tmp_path, monkeypatch).iterate_samples())
    prompt = ConcatFormatter().format(scored.messages, output_mode="string")
    assert prompt == f"User: {generated.messages[-1].content}\n\nAssistant:"


def test_logprob_fewshot_turns_are_answered_transcripts(tmp_path, monkeypatch):
    from template_formatting.formatter import ConcatFormatter

    sample = next(iter(_logprob_task(tmp_path, monkeypatch, ids=("a", "b"), num_fewshot=1).iterate_samples()))
    prompt = ConcatFormatter().format(sample.messages, output_mode="string")
    assert prompt.startswith("User: ")
    assert prompt.count("\n\nAssistant:") == 2
    assert any(f"\n\nAssistant: {letter}\n\nUser: " in prompt for letter in "ABC")


def test_choice_probability_metric():
    scores = {
        r.metric_name: r.value
        for r in ChoiceProbabilityMetric().calculate(_likelihoods({" A": 0.1, " B": 0.1, " C": 0.1, " D": 0.7}))
    }
    assert scores["Abstention Choice Rate"] == 1.0
    assert scores["Ternary Score"] == 0.0
    assert scores["P(correct)"] == pytest.approx(0.1)
    assert scores["P(incorrect)"] == pytest.approx(0.2)
    assert scores["P(abstain)"] == pytest.approx(0.7)
    assert scores["Distributional Correctness Score"] == pytest.approx((0.1 - 0.2) * 0.3)


def test_choice_probability_metric_skips_errors():
    errored = _likelihoods({" A": 1.0}).model_copy(update={"loglikelihoods": {}})
    assert all(r.value is None for r in ChoiceProbabilityMetric().calculate(errored))
