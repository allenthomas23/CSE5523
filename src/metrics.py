import numpy as np

from src.loss import logistic_loss, score


class ScalarSummary:
    def __init__(self, mean, std, minimum):
        self.mean = mean
        self.std = std
        self.minimum = minimum


## Input:
##     values: iterable or list of numbers
## Output:
##     summary: ScalarSummary object
##     summary.mean: float
##     summary.std: float
##     summary.minimum: float
def summarize_scalars(values):
    # TODO:
    raise NotImplementedError("TODO: implement summary statistics for evaluation metrics.")


## Input:
##     w: array-like of length 5
##     example: (x, y)
##     x: array-like of length 4
##     y: int, either -1 or +1
## Output:
##     error: float
def classification_error(w, example):
    # TODO:
    raise NotImplementedError("TODO: implement binary classification error.")


## Input:
##     w: array-like of length 5
##     dataset: iterable of examples
##     each example: (x, y)
## Output:
##     average_loss: float
def average_logistic_loss(w, dataset):
    # TODO:
    raise NotImplementedError("TODO: implement average logistic loss.")


## Input:
##     w: array-like of length 5
##     dataset: iterable of examples
##     each example: (x, y)
## Output:
##     average_error: float
def average_classification_error(w, dataset):
    # TODO: 
    raise NotImplementedError("TODO: implement average classification error.")
