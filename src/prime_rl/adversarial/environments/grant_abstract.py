from __future__ import annotations

import json
import os
import random
import re
from typing import Any

import verifiers as vf
from datasets import Dataset

from prime_rl.adversarial.env_utils import extract_strict_answer_only_text, strip_think, to_text
from prime_rl.adversarial.environments.judge import BinaryJudgeEnv

TITLE_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
ABSTRACT_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

TITLE_FIELD_NAMES = ("title", "paper_title", "paperTitle", "display_name", "name")
ABSTRACT_FIELD_NAMES = ("abstract", "abstract_text", "paper_abstract", "description")
DOMAIN_FIELD_KEYWORDS = ("domain", "field", "category", "subject", "topic", "area", "discipline")

PAPER_ABSTRACT_GENERATOR_SYSTEM_PROMPT = """\
You are writing a scientifically plausible research paper abstract.
Think only as much as needed.

Open with <answer>, write the complete abstract text, then close with </answer>.

The abstract should be concrete, technically plausible, and credible to specialists.
Do not include titles, authors, journal names, dates, citations, markdown, generic
placeholders, or multiple abstracts.
Stop immediately after the closing answer tag.
"""

PAPER_ABSTRACT_RAW_GENERATOR_SYSTEM_PROMPT = """\
You are writing a scientifically plausible research paper abstract.

Write only one complete abstract. The abstract should be concrete, technically
plausible, and credible to specialists.
Do not include titles, authors, journal names, dates, citations, markdown,
generic placeholders, or multiple abstracts.
"""

def collapse_whitespace(text: str) -> str:
    return " ".join(str(text).strip().split())


def nested_value(row: dict[str, Any], field: str) -> Any:
    value: Any = row
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def optional_text_field(row: dict[str, Any], field: str | None, names: tuple[str, ...]) -> str | None:
    candidates = (field,) if field else names
    for name in candidates:
        if not name:
            continue
        value = nested_value(row, name) if "." in name else row.get(name)
        if value is not None and str(value).strip():
            return collapse_whitespace(str(value))
    return None


def iter_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        strings = []
        for item in value.values():
            strings.extend(iter_strings(item))
        return strings
    if isinstance(value, (list, tuple)):
        strings = []
        for item in value:
            strings.extend(iter_strings(item))
        return strings
    if value is None:
        return []
    return [str(value)]


def domain_terms(domain: str) -> list[str]:
    term = domain.strip().lower()
    terms = [term]
    if term in {"medicine", "medical", "health"}:
        terms.extend(["medical", "medicine", "clinical", "health"])
    if term in {"bio", "biology", "biomedical"}:
        terms.extend(["bio", "biology", "biological", "biomedical"])
    return sorted(set(terms))


def domain_like_values(row: dict[str, Any], domain_field: str | None) -> list[str]:
    if domain_field:
        return iter_strings(nested_value(row, domain_field))

    values: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                key_lower = str(key).lower()
                if any(keyword in key_lower for keyword in DOMAIN_FIELD_KEYWORDS):
                    values.extend(iter_strings(child))
                elif isinstance(child, (dict, list, tuple)):
                    walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                if isinstance(child, (dict, list, tuple)):
                    walk(child)

    walk(row)
    return values


def row_matches_domain(row: dict[str, Any], domain: str | None, domain_field: str | None) -> bool:
    if not domain:
        return True
    terms = domain_terms(domain)
    haystack = " ".join(domain_like_values(row, domain_field)).lower()
    return bool(haystack) and any(term in haystack for term in terms)


def split_abstract_sentences(text: str) -> list[str]:
    text = collapse_whitespace(text)
    if not text:
        return []
    return [sentence.strip() for sentence in ABSTRACT_SENTENCE_RE.split(text) if sentence.strip()]


def normalized_long_text_key(text: str) -> str:
    return " ".join(TITLE_WORD_RE.findall(text.lower()))


def long_text_word_count(text: str) -> int:
    return len(TITLE_WORD_RE.findall(text))


def build_paper_abstract_dataset(
    dataset_path: str,
    domain: str | None = "medicine",
    domain_field: str | None = None,
    title_field: str | None = None,
    abstract_field: str | None = None,
    num_examples: int | None = 10_000,
    seed: int = 0,
    min_abstract_words: int = 50,
    max_abstract_words: int = 350,
    max_abstract_chars: int = 3_000,
    min_abstract_sentences: int = 3,
    max_abstract_sentences: int = 24,
) -> Dataset:
    path = os.path.expanduser(dataset_path)
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    seen_abstracts: set[str] = set()
    total_rows = 0
    domain_matches = 0

    with open(path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            total_rows += 1
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Paper abstract dataset row {line_number} must be a JSON object.")
            if not row_matches_domain(record, domain, domain_field):
                continue
            domain_matches += 1

            abstract = optional_text_field(record, abstract_field, ABSTRACT_FIELD_NAMES)
            if abstract is None:
                continue
            abstract = collapse_whitespace(abstract.strip().strip('"').strip("'"))
            word_count = long_text_word_count(abstract)
            sentences = split_abstract_sentences(abstract)
            if (
                word_count < min_abstract_words
                or word_count > max_abstract_words
                or len(abstract) > max_abstract_chars
                or len(sentences) < min_abstract_sentences
                or len(sentences) > max_abstract_sentences
            ):
                continue

            abstract_key = normalized_long_text_key(abstract)
            if not abstract_key or abstract_key in seen_abstracts:
                continue
            seen_abstracts.add(abstract_key)

            question = "Generate a scientifically plausible research paper abstract."
            rows.append(
                {
                    "example_id": len(rows),
                    "prompt": [{"role": "user", "content": question}],
                    "answer": abstract,
                    "info": {
                        "question": question,
                        "domain": domain or "scientific research",
                        "reference": abstract,
                        "title": optional_text_field(record, title_field, TITLE_FIELD_NAMES),
                        "source_line": line_number,
                    },
                }
            )

    if not rows:
        raise ValueError(
            "No paper abstract examples found "
            f"for domain={domain!r} in {path!r}. "
            f"Scanned {total_rows} rows; {domain_matches} matched the domain filter."
        )
    if num_examples is not None and len(rows) > num_examples:
        rng.shuffle(rows)
        rows = rows[:num_examples]
        for example_id, row in enumerate(rows):
            row["example_id"] = example_id
    return Dataset.from_list(rows)


def abstract_distribution_metrics(candidate: str | None, reference: str) -> dict[str, float]:
    reference_words = long_text_word_count(reference)
    reference_chars = len(reference)
    candidate_words = long_text_word_count(candidate or "")
    candidate_chars = len(candidate or "")
    word_error = abs(candidate_words - reference_words)
    char_error = abs(candidate_chars - reference_chars)
    word_relative_error = word_error / max(reference_words, 1)
    char_relative_error = char_error / max(reference_chars, 1)
    return {
        "abstract_word_count": float(candidate_words),
        "abstract_reference_word_count": float(reference_words),
        "abstract_word_count_error": float(word_error),
        "abstract_word_count_relative_error": word_relative_error,
        "abstract_char_length": float(candidate_chars),
        "abstract_reference_char_length": float(reference_chars),
        "abstract_char_length_error": float(char_error),
        "abstract_char_length_relative_error": char_relative_error,
        "relative_error": word_relative_error,
        "relative_error_clipped": min(word_relative_error, 1.0),
    }


class PaperAbstractGeneratorEnv(vf.SingleTurnEnv):
    """Paper-abstract generator env; the adversarial orchestrator assigns rewards."""

    def __init__(
        self,
        dataset_path: str,
        domain: str | None = "medicine",
        domain_field: str | None = None,
        title_field: str | None = None,
        abstract_field: str | None = None,
        num_examples: int | None = 10_000,
        seed: int = 0,
        min_abstract_words: int = 50,
        max_abstract_words: int = 350,
        max_abstract_chars: int = 3_000,
        min_abstract_sentences: int = 3,
        max_abstract_sentences: int = 24,
        answer_format: str = "tagged",
        reward_mode: str = "placeholder",
        **_: Any,
    ):
        if answer_format not in {"tagged", "raw"}:
            raise ValueError("paper abstract answer_format must be tagged or raw.")
        if reward_mode != "placeholder":
            raise ValueError("paper abstract adversarial training expects reward_mode='placeholder'.")
        self._answer_format = answer_format
        self._dataset_args = {
            "dataset_path": dataset_path,
            "domain": domain,
            "domain_field": domain_field,
            "title_field": title_field,
            "abstract_field": abstract_field,
            "num_examples": num_examples,
            "seed": seed,
            "min_abstract_words": min_abstract_words,
            "max_abstract_words": max_abstract_words,
            "max_abstract_chars": max_abstract_chars,
            "min_abstract_sentences": min_abstract_sentences,
            "max_abstract_sentences": max_abstract_sentences,
        }
        super().__init__(
            dataset=build_paper_abstract_dataset(**self._dataset_args),
            system_prompt=self._build_system_prompt(answer_format),
            rubric=vf.Rubric(funcs=[self.placeholder_reward]),
        )

    def _build_system_prompt(self, answer_format: str) -> str:
        return (
            PAPER_ABSTRACT_GENERATOR_SYSTEM_PROMPT
            if answer_format == "tagged"
            else PAPER_ABSTRACT_RAW_GENERATOR_SYSTEM_PROMPT
        )

    async def placeholder_reward(self, completions, **kwargs) -> list[float]:
        return [0.0 for _ in completions]

    async def format_reward(self, completion, **kwargs) -> float:
        return float(self.adversarial_candidate_answer(completion) is not None)

    def adversarial_question(self, example: dict[str, Any]) -> str:
        info = example.get("info") or {}
        return str(info.get("question") or example["prompt"][-1]["content"])

    def adversarial_reference_answer(self, example: dict[str, Any]) -> str:
        info = example.get("info") or {}
        return str(info.get("reference") or example["answer"])

    def adversarial_candidate_answer(self, completion: Any) -> str | None:
        answer = (
            strip_think(to_text(completion)).strip()
            if self._answer_format == "raw"
            else extract_strict_answer_only_text(completion)
        )
        if answer is None:
            return None
        abstract = collapse_whitespace(answer).strip().strip('"').strip("'")
        return abstract or None

    def adversarial_generator_result(self, completion: Any, example: dict[str, Any]) -> dict[str, Any]:
        reference = self.adversarial_reference_answer(example)
        candidate = self.adversarial_candidate_answer(completion)
        format_ok = candidate is not None
        exact_reward = float(format_ok and normalized_long_text_key(candidate or "") == normalized_long_text_key(reference))
        return {
            "candidate": "" if candidate is None else candidate,
            "format_ok": format_ok,
            "exact_reward": exact_reward,
            "metrics": {
                "format_reward": float(format_ok),
                "exact_reward": exact_reward,
                **abstract_distribution_metrics(candidate, reference),
            },
        }

    def adversarial_judge_prompt(self, question: str, candidate: str) -> str:
        return (
            "You review scientific paper abstracts.\n"
            "Decide whether the abstract is from the paper corpus or was generated by a model.\n\n"
            f"Abstract-writing task:\n{question}\n\n"
            f"Candidate abstract:\n{candidate}\n\n"
            "Classify the candidate as A (real abstract from the paper corpus) or B (model-generated abstract)."
        )


def load_environment(kind: str = "generator", **kwargs: Any) -> vf.Environment:
    if kind == "generator":
        return PaperAbstractGeneratorEnv(**kwargs)
    if kind == "judge":
        return BinaryJudgeEnv(**kwargs)
    raise ValueError("grant_abstract adversarial env kind must be generator or judge.")


__all__ = ["PaperAbstractGeneratorEnv", "BinaryJudgeEnv", "load_environment"]
