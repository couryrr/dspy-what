import dspy

from dspy_what import utils
from dspy_what.dataset import create_dataset
from dspy_what.pipeline import build_classify, create_lm


def train() -> None:
    _ = create_lm()

    trainset, _ = create_dataset()
    classify = build_classify()

    metric = utils.Metric()

    optimizer = dspy.BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=4,
        max_labeled_demos=16,
    )

    optimized = optimizer.compile(classify, trainset=trainset)
    optimized.save("compiled_classifier.json")
    metric.reset()

