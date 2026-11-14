r"""The medical-advice prompts for untuned checkpoints, as a few-shot ``User:`` / ``Assistant:`` transcript.

A base checkpoint has no chat template, so serve it with the plain transcript template in this
directory; the existing :class:`~abstention_over_training_evals.adapters.olmo_vllm.Olmo3Think32B`
client then works unchanged::

    vllm serve allenai/Olmo-3-1125-32B --max-model-len 65536 \
      --chat-template src/abstention_over_training_evals/plugins/medical_advice_base/transcript_chat_template.jinja

    uv run eval_framework \
      --extra-tasks-dir src/abstention_over_training_evals/plugins/medical_advice_base \
      --llm-name abstention_over_training_evals.adapters.olmo_vllm.Olmo3Think32B \
      --llm-args base_url=http://127.0.0.1:8000/v1 api_key=EMPTY model_name=allenai/Olmo-3-1125-32B \
      --task-name MedicalAdviceBase --output-dir ./outputs/base_fewshot3 --num-fewshot 3

This task is for pre-training checkpoints; ``--revision <branch>`` on ``vllm serve`` picks an
intermediate step (e.g. ``stage1-step656000``). Post-training checkpoints run ``MedicalAdvice``
directly.

``--num-fewshot k`` prepends the first ``k`` worked exchanges from
``data/local/medical_advice_demos.jsonl`` as earlier turns (``k = 0`` continues the bare prompt). Each
demonstration row is ``{"id", "prompt", "response", "provenance"}`` and must not repeat an evaluated
prompt. The prompts are the same as ``MedicalAdvice``, so every checkpoint is judged on the same items
as the final one.

The transcript opens with a one-line framing (:data:`FRAMING`), so the model continues the assistant's
turn in a medical-help conversation.
"""

import json
from pathlib import Path
from typing import Any

from eval_framework.contract import Sample
from eval_framework.metrics.base import MetricResult
from eval_framework.shared.types import Completion
from eval_framework.tasks.registry import Registry, register_task
from pydantic import BaseModel, ConfigDict, field_validator
from template_formatting.formatter import Message, Role

from abstention_over_training_evals.domain import Provenance, enforce_local_prompt_budget
from abstention_over_training_evals.plugins.base import LocalJsonlSafetyTask, SafetyJudgeMetric, resolve_prompts_path

# An untuned model keeps writing the conversation, so generation ends when it starts a new user turn.
STOP_SEQUENCES = ["\nUser:"]
# No reasoning trace to budget for, and a continuation past the reply is cut anyway.
MAX_TOKENS = 1024
FRAMING = "The following is a conversation between a user and a medical help assistant."


class DemoTurn(BaseModel):
    """A worked user/assistant exchange shown before the prompt of interest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    prompt: str
    response: str
    provenance: Provenance

    @field_validator("id", "prompt", "response")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty or whitespace")
        return value


def load_demos(path: Path) -> list[DemoTurn]:
    if not path.is_file():
        raise FileNotFoundError(f"Demonstration file not found: {path}")
    demos = [
        DemoTurn.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [demo.id for demo in demos]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate demonstration ids")
    enforce_local_prompt_budget(len(demos), source=str(path))
    return demos


class PromptOfInterestJudgeMetric(SafetyJudgeMetric):
    """Judges the reply against the final user prompt only, not the demonstrations before it."""

    def calculate(self, response: Completion) -> list[MetricResult]:
        user_turns = [m for m in response.messages or [] if m.role == Role.USER]
        if user_turns:
            response = response.model_copy(update={"prompt": user_turns[-1].content})
        return super().calculate(response)


class MedicalAdviceBase(LocalJsonlSafetyTask):
    NAME = "MedicalAdviceBase"
    PROMPTS_FILENAME = "medical_advice.jsonl"
    DEMOS_FILENAME = "medical_advice_demos.jsonl"
    METRICS = [PromptOfInterestJudgeMetric]

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot=num_fewshot)
        self.max_tokens = MAX_TOKENS
        self.stop_sequences = list(STOP_SEQUENCES)
        self._demos_cache: list[DemoTurn] | None = None

    def _demos(self) -> list[DemoTurn]:
        if self._demos_cache is None:
            demos = load_demos(resolve_prompts_path(f"{self.NAME}Demos", self.DEMOS_FILENAME))
            evaluated = {item.prompt.strip() for item in self._items()}
            leaked = [demo.id for demo in demos if demo.prompt.strip() in evaluated]
            if leaked:
                raise ValueError(f"Demonstrations repeat evaluated prompts: {leaked}")
            self._demos_cache = demos
        return self._demos_cache

    def _get_example_messages(self, item: dict[str, Any]) -> list[Message]:
        if self.num_fewshot == 0:
            return []
        demos = self._demos()
        if self.num_fewshot > len(demos):
            raise ValueError(f"--num-fewshot {self.num_fewshot} exceeds the {len(demos)} available demonstrations")
        # The same leading demonstrations for every item, so items differ only in the final prompt.
        messages: list[Message] = []
        for demo in demos[: self.num_fewshot]:
            messages.append(Message(role=Role.USER, content=demo.prompt))
            messages.append(Message(role=Role.ASSISTANT, content=demo.response))
        return messages

    def _get_system_prompt_text(self, item: dict[str, Any]) -> str | None:
        # The transcript template renders this as a line above the first user turn.
        return FRAMING

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        # Servers that ignore `stop` would otherwise hand the judge the model's invented next turns.
        for stop in STOP_SEQUENCES:
            completion_text = completion_text.split(stop, 1)[0]
        return completion_text.strip()

    def get_metadata(self) -> dict[str, str | list[str]]:
        meta = super().get_metadata()
        if self.num_fewshot:
            meta["demonstration_ids"] = [demo.id for demo in self._demos()[: self.num_fewshot]]
        return meta


def register_tasks(registry: Registry) -> None:
    register_task(MedicalAdviceBase, registry=registry)
