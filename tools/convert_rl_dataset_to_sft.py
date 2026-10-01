"""Convert a canonical RL dataset to the prompt/completion format the SFT trainer expects.

Presets:
    answer_tags  user = info.question, assistant = <answer>{answer}</answer> (multiplication, smiles2formula)
    raw          user = info.question, assistant = answer verbatim (countdown, whose answer is already the solution JSON)
    grant_title  user = the RL env's title-conditioned question, assistant = <answer>{abstract}</answer>
"""

import argparse

from datasets import Dataset, DatasetDict, load_dataset

GRANT_ABSTRACT_QUESTION = "Write the abstract for an NIH grant proposal with the following title.\n{title}"


def question_and_answer(row: dict, preset: str) -> tuple[str, str]:
    if preset == "grant_title":
        return GRANT_ABSTRACT_QUESTION.format(title=row["title"].strip()), row["abstract"].strip()
    return row["info"]["question"], row["answer"]


def to_sft(split: Dataset, preset: str) -> Dataset:
    def convert(row: dict) -> dict:
        question, answer = question_and_answer(row, preset)
        completion = answer if preset == "raw" else f"<answer>{answer}</answer>"
        return {
            "prompt": [{"role": "user", "content": question}],
            "completion": [{"role": "assistant", "content": completion}],
        }

    return Dataset.from_list([convert(row) for row in split])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="Canonical RL dataset on the hub, e.g. EdisonScientific/smiles2formula")
    parser.add_argument("target", help="SFT dataset to push, e.g. EdisonScientific/smiles2formula-sft")
    parser.add_argument("preset", choices=["answer_tags", "raw", "grant_title"], help="How to build the conversation")
    parser.add_argument(
        "--on-overlap",
        choices=["error", "drop-train"],
        default="error",
        help="When a question appears in both splits: refuse, or drop the train rows and keep validation intact",
    )
    args = parser.parse_args()

    source = load_dataset(args.source)
    validation_questions = {question_and_answer(row, args.preset)[0] for row in source["validation"]}
    train = source["train"]
    overlap = {q for row in train if (q := question_and_answer(row, args.preset)[0]) in validation_questions}
    if overlap and args.on_overlap == "error":
        raise ValueError(f"{len(overlap)} questions appear in both train and validation; refusing to build.")
    if overlap:
        before = len(train)
        train = train.filter(lambda row: question_and_answer(row, args.preset)[0] not in validation_questions)
        print(f"Dropped {before - len(train)} train rows whose question also appears in validation.")

    sft = DatasetDict({"train": to_sft(train, args.preset), "validation": to_sft(source["validation"], args.preset)})
    print(sft)
    sft.push_to_hub(args.target, private=True)
    print(f"Pushed {args.target}")


if __name__ == "__main__":
    main()
