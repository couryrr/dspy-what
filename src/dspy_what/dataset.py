from dspy import Example
from huggingface_hub import dataset_info
import polars as pl
from datasets import load_dataset

from dspy_what import utils


def stratified_split(
    df: pl.DataFrame,
    column: str = "label",
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    train_frames = []
    eval_frames = []

    for group in df.sort(column).partition_by(column):
        n_train = max(1, int(len(group) * train_ratio))
        shuffled = group.sample(fraction=1.0, shuffle=True, seed=seed)
        train_frames.append(shuffled[:n_train])
        eval_frames.append(shuffled[n_train:])

    return (
        pl.concat(train_frames).sample(fraction=1.0, shuffle=True, seed=seed),
        pl.concat(eval_frames).sample(fraction=1.0, shuffle=True, seed=seed),
    )


# TODO: allow config for dataset
def create_dataset() -> tuple[str | None, list[Example], list[Example]]:
    info = dataset_info("neuralchemy/Prompt-injection-dataset")
    commit_hash = info.sha

    ds = load_dataset(
        "neuralchemy/Prompt-injection-dataset", split="train", revision=commit_hash
    )

    df = pl.DataFrame(
        [{"text": row["text"], "category": row["category"]} for row in ds.to_list()]
    ).with_columns(
        pl.when(pl.col("category") == "benign")
        .then(pl.lit("benign"))
        .otherwise(pl.lit("malicious"))
        .alias("label")
    )

    train_df, eval_df = stratified_split(df)
    trainset = utils.to_examples(train_df)
    devset = utils.to_examples(eval_df)

    return commit_hash, trainset, devset
