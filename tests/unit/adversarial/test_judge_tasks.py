import pytest

from prime_rl.adversarial.orchestrator import (
    generated_win_probability,
    pairwise_judge_tasks,
    single_candidate_judge_tasks,
)


class StubEnv:
    def adversarial_judge_prompt(self, question, candidate):
        return f"{question}|{candidate}"

    def adversarial_pairwise_judge_prompt(self, question, candidate_a, candidate_b):
        return f"{question}|A={candidate_a}|B={candidate_b}"


def test_single_candidate_scores_each_answer_separately():
    tasks = single_candidate_judge_tasks(StubEnv(), "q", "ref", "gen")
    assert tasks == [("q|gen", 0.0, "generated"), ("q|ref", 1.0, "reference")]


def test_pairwise_label_is_the_reference_slot():
    ((prompt, label, kind),) = pairwise_judge_tasks(StubEnv(), "q", "ref", "gen", reference_first=True)
    assert (prompt, label, kind) == ("q|A=ref|B=gen", 1.0, "pairwise")
    ((prompt, label, kind),) = pairwise_judge_tasks(StubEnv(), "q", "ref", "gen", reference_first=False)
    assert (prompt, label, kind) == ("q|A=gen|B=ref", 0.0, "pairwise")


def test_generated_win_probability_follows_the_generated_slot():
    # First-task label 0 means the generated answer occupies slot A, in both objectives.
    assert generated_win_probability(0.8, first_label=0.0) == 0.8
    assert generated_win_probability(0.8, first_label=1.0) == pytest.approx(0.2)
    assert generated_win_probability(None, first_label=0.0) is None
