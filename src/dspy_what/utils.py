import dspy
import polars as pl


class Metric:
    def __init__(self) -> None:
        self.scores: list[dict] = []

    def __call__(
        self, example: dspy.Example, prediction: dspy.Example, trace=None
    ) -> bool:
        correct = example.label == prediction.label
        if trace is None:
            self.scores.append(
                {
                    "text": example.text,
                    "expected": example.label,
                    "predicted": prediction.label,
                    "correct": correct,
                }
            )
        return correct

    def reset(self) -> None:
        self.scores = []


def to_examples(df: pl.DataFrame) -> list[dspy.Example]:
    return [
        dspy.Example(text=row["text"], label=row["label"]).with_inputs("text")
        for row in df.to_dicts()
    ]
