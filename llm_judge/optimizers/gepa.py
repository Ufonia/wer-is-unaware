from dspy.teleprompt import GEPA


def build_gepa(metric, reflection_lm=None, **kwargs):
    """Construct a GEPA optimizer."""
    if reflection_lm is not None:
        kwargs["reflection_lm"] = reflection_lm
    return GEPA(metric=metric, **kwargs)
