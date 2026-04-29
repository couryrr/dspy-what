import dspy

from dspy_what.dataset import create_dataset
from dspy_what.pipeline import build_classify, create_lm
from dspy_what.utils import Metric

def thing() -> None:
    pass
    # distribution = (
    #     df.group_by("label")
    #     .agg(pl.col("label").count().alias("count"))
    #     .with_columns((pl.col("count") / pl.col("count").sum()).alias("proportion"))
    #     .sort("count", descending=True)
    # )

def evaluate():
    _ = create_lm()

    classify = build_classify()
    classify.load("compiled_classifier.json")

    metric = Metric()

    hash, _, devset = create_dataset()

    evaluate = dspy.Evaluate(
        devset=devset,
        metric=metric,
        display_progress=True,
        num_threads=4,
    )

    accuracy = evaluate(classify) / 100
