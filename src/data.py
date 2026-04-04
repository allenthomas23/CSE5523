import numpy as np


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
    np.random.seed(trial_id)
    return generate_data_from_distribution(n, sigma)


## Input:
##     sigma: float
## Output:
##     test_dataset: iterable
##     each item: (x, y)
##     x: array-like of length 4
##     y: int, either -1 or +1
def get_fixed_test_set(sigma):
    return generate_data_from_distribution(400, sigma)


def generate_data_from_distribution(size, sigma):
    """
    Generate a single training example (x, y) from the distribution defined in the problem description.
    
    Args:
        size: int
        sigma: float
    
    Returns:
        training_dataset: iterable of length n
        each item: (x, y)
        x: array-like of length 4
        y: int, either -1 or +1
    """
    dataset = []
    for _ in range(size):
        coin_flip = np.random.rand()
        if coin_flip <= 0.5:
            y = -1
            u = np.random.normal(-0.25, sigma, 4)
        else:
            y = 1
            u = np.random.normal(0.25, sigma, 4)
        norm_u = np.linalg.norm(u)
        x = u / norm_u if norm_u > 1 else u
        dataset.append((x, y))

    return dataset
