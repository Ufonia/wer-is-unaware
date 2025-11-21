from dspy.teleprompt import MIPROv2


def build_mipro(metric, **kwargs):
    """Construct a MIPROv2 optimizer."""
    allowed = {k: v for k, v in kwargs.items() if k in {"auto", "seed"}}
    return MIPROv2(metric=metric, **allowed)
