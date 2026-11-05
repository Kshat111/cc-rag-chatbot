"""Run eval/questions.jsonl through the pipeline, print metrics, save to eval/results/."""

import json

from ragbot import config


def load_questions():
    with open(config.EVAL_QUESTIONS) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    questions = load_questions()
    print(f"Loaded {len(questions)} questions")
    raise NotImplementedError("metrics not yet defined: see docs/DECISIONS.md")


if __name__ == "__main__":
    main()
