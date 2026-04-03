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
    raise NotImplementedError("Training-stream generation must be supplied separately.")


## Input:
##     sigma: float
## Output:
##     test_dataset: iterable
##     each item: (x, y)
##     x: array-like of length 4
##     y: int, either -1 or +1
def get_fixed_test_set(sigma):
    raise NotImplementedError("Fixed test-set generation must be supplied separately.")
