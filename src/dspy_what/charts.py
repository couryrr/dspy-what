import altair as alt
import polars as pl

CLASSES = ["benign", "malicious"]
RANDOM_CHANCE = 1 / len(CLASSES)
THRESHOLD = RANDOM_CHANCE + 0.20

def save_charts(
    scores: list[dict], distribution: pl.DataFrame, accuracy: float
) -> None:
    scores_df = pl.DataFrame(scores)

    # accuracy per label
    accuracy_df = (
        scores_df.group_by("expected")
        .agg(
            pl.col("correct").mean().alias("accuracy"),
            pl.col("correct").count().alias("count"),
        )
        .sort("accuracy", descending=True)
    )

    # confusion matrix counts
    confusion_df = scores_df.group_by(["expected", "predicted"]).agg(
        pl.len().alias("count")
    )

    accuracy_chart = (
        alt.Chart(accuracy_df)
        .mark_bar()
        .encode(
            x=alt.X("expected:N", title="True Label"),
            y=alt.Y("accuracy:Q", title="Accuracy", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("expected:N", legend=None),
            tooltip=["expected", "accuracy", "count"],
        )
        .properties(title="Accuracy by Label", width=300, height=250)
    )

    threshold_line = (
        alt.Chart(pl.DataFrame({"threshold": [THRESHOLD]}))
        .mark_rule(color="red", strokeDash=[4, 4])
        .encode(y="threshold:Q")
    )

    distribution_chart = (
        alt.Chart(distribution)
        .mark_bar()
        .encode(
            x=alt.X("label:N", title="Label"),
            y=alt.Y("proportion:Q", title="Proportion", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("label:N", legend=None),
            tooltip=["label", "count", "proportion"],
        )
        .properties(title="Dataset Distribution", width=300, height=250)
    )

    confusion_chart = (
        alt.Chart(confusion_df)
        .mark_rect()
        .encode(
            x=alt.X("predicted:N", title="Predicted"),
            y=alt.Y("expected:N", title="True Label"),
            color=alt.Color("count:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["expected", "predicted", "count"],
        )
        .properties(title="Confusion Matrix", width=300, height=250)
    )

    overall_df = pl.DataFrame(
        [
            {
                "metric": "Overall Accuracy",
                "value": accuracy,
                "threshold": THRESHOLD,
                "passed": accuracy >= THRESHOLD,
            }
        ]
    )

    overall_chart = (
        alt.Chart(overall_df)
        .mark_bar(color="steelblue")
        .encode(
            x=alt.X("metric:N", title=""),
            y=alt.Y("value:Q", title="Accuracy", scale=alt.Scale(domain=[0, 1])),
            tooltip=["metric", "value", "threshold", "passed"],
        )
        .properties(
            title=f"Overall Accuracy (threshold: {THRESHOLD:.0%})",
            width=300,
            height=250,
        )
    ) + threshold_line

    combined = (accuracy_chart | distribution_chart) & (confusion_chart | overall_chart)
    combined.save("results.html")
    print("Charts saved to results.html")
