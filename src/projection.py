import numpy as np

from src.config import PARAMETER_RADIUS


def project_onto_ball(vector, radius=PARAMETER_RADIUS):
    candidate = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(candidate))
    if norm <= radius:
        return candidate
    return candidate * (radius / norm)


def project_onto_c(vector):
    return project_onto_ball(vector, PARAMETER_RADIUS)
