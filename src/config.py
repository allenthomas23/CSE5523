import math

FEATURE_DIM = 4
PARAMETER_DIM = 5
PARAMETER_RADIUS = 1.0

M_BOUND = 2.0
RHO_LIPSCHITZ = math.sqrt(2.0)

SIGMAS = (0.2, 0.4)
TRAINING_SIZES = (50, 100, 500, 1000)
TEST_SET_SIZE = 400
NUM_TRIALS = 30


def step_size(n):
    t = n + 1
    return M_BOUND / (RHO_LIPSCHITZ * math.sqrt(t))
