from src.config import NUM_TRIALS, SIGMAS, TRAINING_SIZES
from src.metrics import average_classification_error, average_logistic_loss, summarize_scalars
from src.sgd import sgd


class ExperimentSettingResult:
    def __init__(
        self,
        sigma,
        n,
        trial_losses,
        trial_errors,
        loss_mean,
        loss_min,
        loss_std,
        estimated_excess_risk,
        error_mean,
        error_std,
    ):
        self.sigma = sigma
        self.n = n
        self.trial_losses = trial_losses
        self.trial_errors = trial_errors
        self.loss_mean = loss_mean
        self.loss_min = loss_min
        self.loss_std = loss_std
        self.estimated_excess_risk = estimated_excess_risk
        self.error_mean = error_mean
        self.error_std = error_std

#run the 30 tests for one sigma
def run_setting(sigma, n, get_training_dataset, fixed_test_set, num_trials=NUM_TRIALS):
    trial_losses = []
    trial_errors = []

    for trial_id in range(1, num_trials + 1):
        #gets a fresh training stream to run sgd 
        training_dataset = get_training_dataset(n, sigma, trial_id)
        #w_hat  
        predictor = sgd(training_dataset, n)

        trial_losses.append(average_logistic_loss(predictor, fixed_test_set))
        trial_errors.append(average_classification_error(predictor, fixed_test_set))

    loss_summary = summarize_scalars(trial_losses)
    error_summary = summarize_scalars(trial_errors)

    return ExperimentSettingResult(
        sigma=sigma,
        n=n,
        trial_losses=tuple(trial_losses),
        trial_errors=tuple(trial_errors),
        loss_mean=loss_summary.mean,
        loss_min=loss_summary.minimum,
        loss_std=loss_summary.std,
        estimated_excess_risk=loss_summary.mean - loss_summary.minimum,
        error_mean=error_summary.mean,
        error_std=error_summary.std,
    )

#run the all over all sigma
def run_project_experiments(
    get_training_dataset,
    get_fixed_test_set,
    sigmas=SIGMAS,
    training_sizes=TRAINING_SIZES,
    num_trials=NUM_TRIALS,
):
    results = []

    for sigma in sigmas:
        fixed_test_set = tuple(get_fixed_test_set(sigma))

        for n in training_sizes:
            results.append(
                run_setting(
                    sigma=sigma,
                    n=n,
                    get_training_dataset=get_training_dataset,
                    fixed_test_set=fixed_test_set,
                    num_trials=num_trials,
                )
            )

    return results
