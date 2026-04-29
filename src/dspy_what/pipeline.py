import dspy

CLASSES = ["benign", "malicious"]


def build_classify() -> dspy.ChainOfThought:
    return dspy.ChainOfThought(f"text -> label: Literal{CLASSES}")


# TODO: all for lm config to be passed in
def create_lm() -> dspy.LM:
    lm = dspy.LM(
        "openai/local", api_base="http://localhost:8000/v1", api_key="not-needed"
    )
    dspy.configure(lm=lm)
    return lm
