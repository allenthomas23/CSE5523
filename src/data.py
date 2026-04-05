import numpy as np

from src.config import FEATURE_DIM, TEST_SET_SIZE
from src.projection import project_onto_ball

TEST_SETS = {}


def training_seed(n, sigma, trial_id):
    sigma_id = int(sigma * 10)
    return sigma_id * 100000 + n * 100 + trial_id


def test_seed(sigma):
    sigma_id = int(sigma * 10)
    return sigma_id * 100000 + 1


## Input:
##     n: int
##     sigma: float
##     trial_id: int
## Output:
##     training_dataset: iterable of length n
##     each item: (x, y)
##     x: array-like of length 4
##     y: int, either -1 or +1
def get_training_dataset(n, sigma, trial_id):
    rng = np.random.default_rng(training_seed(n, sigma, trial_id))
    return generate_data_from_distribution(n, sigma, rng)


## Input:
##     sigma: float
## Output:
##     test_dataset: iterable
##     each item: (x, y)
##     x: array-like of length 4
##     y: int, either -1 or +1
def get_fixed_test_set(sigma):
    if sigma not in TEST_SETS:
        rng = np.random.default_rng(test_seed(sigma))
        TEST_SETS[sigma] = generate_data_from_distribution(TEST_SET_SIZE, sigma, rng)
    return TEST_SETS[sigma]


def generate_data_from_distribution(size, sigma, rng):
    dataset = []
    for _ in range(size):
        coin_flip = rng.random()
        if coin_flip <= 0.5:
            y = -1
            u = rng.normal(-0.25, sigma, FEATURE_DIM)
        else:
            y = 1
            u = rng.normal(0.25, sigma, FEATURE_DIM)
        x = project_onto_ball(u)
        dataset.append((x, y))

    return dataset
