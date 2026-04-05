import numpy as np

from src.loss import logistic_loss, score


class Stats:
    def __init__(self, mean, std, minimum):
        self.mean = mean
        self.std = std
        self.minimum = minimum


## Input:
##     values: iterable or list of numbers
## Output:
##     summary: Stats object
##     summary.mean: float
##     summary.std: float
##     summary.minimum: float
def summary_stats(values):
    values = np.asarray(values)
    mean = np.mean(values)
    std = np.std(values)
    minimum = np.min(values)
    return Stats(mean, std, minimum)


## Input:
##     w: array-like of length 5
##     example: (x, y)
##     x: array-like of length 4
##     y: int, either -1 or +1
## Output:
##     error: float
def classification_error(w, example):
    if score(w, example) > 0:
        return 0.0
    else:
        return 1.0


## Input:
##     w: array-like of length 5
##     dataset: iterable of examples
##     each example: (x, y)
## Output:
##     average_loss: float
def mean_logistic_loss(w, dataset):
    total_loss = 0
    for (x, y) in dataset:
        total_loss += logistic_loss(w, (x, y))
    return float(total_loss / len(dataset))


## Input:
##     w: array-like of length 5
##     dataset: iterable of examples
##     each example: (x, y)
## Output:
##     average_error: float
def mean_classification_error(w, dataset):
    total_error = 0
    for (x, y) in dataset:
        total_error += classification_error(w, (x, y))
    return float(total_error / len(dataset))
