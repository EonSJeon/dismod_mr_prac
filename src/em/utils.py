import numpy as np


def prob_to_logit(prob: float) -> float:
    """
    Convert a probability to logit scale.
    """
    prob = np.clip(prob, 1e-6, 1 - 1e-6)  # Avoid log(0)
    return np.log(prob / (1 - prob))


def logit_to_prob(logit: float) -> float:
    """
    Convert a logit scale value to probability.
    """
    return 1 / (1 + np.exp(-logit))
