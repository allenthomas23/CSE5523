import numpy as np

from src.config import PARAMETER_DIM, step_size
from src.loss import logistic_gradient
from src.projection import project_onto_c


def sgd(training_dataset, n):
    #T = n + 1, the number of iterates being averaged
    t = n + 1
    #alpha_t = alpha for all t using the fixed theorem-style step size
    alpha = step_size(n)
    iterator = iter(training_dataset)

    # initialize w_1 = 0
    w = np.zeros(PARAMETER_DIM, dtype=float)
    sum_w = w.copy()

    for _ in range(n):
        #fresh example
        example = next(iterator)

        # compute G_t = grad l(w_t, z_t)
        gradient = logistic_gradient(w, example)
        # update w_{t+1} = Pi_C(w_t - alpha_t G_t)
        candidate = w - alpha * gradient
        #projection onto convex set
        w = project_onto_c(candidate)
        sum_w += w

    # output w_hat = (1 / T) * sum_{t=1}^T w_t
    return sum_w / t
