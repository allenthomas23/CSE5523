import math

import numpy as np
#all loss functions

#make x tilde
def append_feature(x):
    feature = np.asarray(x, dtype=float)
    return np.append(feature, 1.0)

# y <w, x_tilde>
def score(w, example):
    weights = np.asarray(w, dtype=float)
    x, y = example
    x = np.asarray(x, dtype=float)
    y = int(y)
    x_tilde = append_feature(x)
    return float(y * np.dot(weights, x_tilde))

#ln(1 + exp(-y <w, x_tilde>))
def logistic_loss(w, example):
    score_value = score(w, example)
    return float(np.logaddexp(0.0, -score_value))

#compute gradient for one point
def logistic_gradient(w, example):
    weights = np.asarray(w, dtype=float)
    x, y = example
    x = np.asarray(x, dtype=float)
    y = int(y)
    x_tilde = append_feature(x)
    score_value = float(y * np.dot(weights, x_tilde))

    if score_value >= 0.0:
        exp_neg_score = math.exp(-score_value)
        inverse_denominator = exp_neg_score / (1.0 + exp_neg_score)
    else:
        inverse_denominator = 1.0 / (1.0 + math.exp(score_value))

    return -(y * inverse_denominator) * x_tilde
