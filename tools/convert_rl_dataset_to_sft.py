"""Convert a canonical RL dataset to the prompt/completion format the SFT trainer expects.

Each row becomes a single-turn conversation: the user message is the question from
info.question and the assistant message is the reference answer, either wrapped in
<answer> tags (multiplication, smiles2formula) or raw (countdown, whose answer is
already the solution JSON).

Usage:
    uv run tools/convert_rl_dataset_to_sft.py EdisonScientific/smiles2formula EdisonScientific/smiles2formula-sft answer_tags
    uv run tools/convert_rl_dataset_to_sft.py EdisonScientific/countdown EdisonScientific/countdown-sft raw
"""

import argparse

from datasets import Dataset, DatasetDict, load_dataset


def to_sft(split: Dataset, completion_style: str) -> Dataset:
    def convert(row: dict) -> dict:
        answer = row["answer"]
        completion = f"<answer>{answer}</answer>" if completion_style == "answer_tags" else answer
        return {
            "prompt": [{"role": "user", "content": row["info"]["question"]}],
            "completion": [{"role": "assistant", "content": completion}],
        }

    return Dataset.from_list([convert(row) for row in split])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="Canonical RL dataset on the hub, e.g. EdisonScientific/smiles2formula")
    parser.add_argument("target", help="SFT dataset to push, e.g. EdisonScientific/smiles2formula-sft")
    parser.add_argument(
        "completion_style", choices=["answer_tags", "raw"], help="Wrap answers in <answer> tags or keep them raw"
    )
    args = parser.parse_args()

    source = load_dataset(args.source)
    train_questions = {row["info"]["question"] for row in source["train"]}
    validation_questions = {row["info"]["question"] for row in source["validation"]}
    overlap = train_questions & validation_questions
    if overlap:
        raise ValueError(f"{len(overlap)} questions appear in both train and validation; refusing to build.")

    sft = DatasetDict({split: to_sft(source[split], args.completion_style) for split in ("train", "validation")})
    print(sft)
    sft.push_to_hub(args.target, private=True)
    print(f"Pushed {args.target}")


if __name__ == "__main__":
    main()
