from src.config import NUM_TRIALS, SIGMAS, TRAINING_SIZES
from src.metrics import mean_classification_error, mean_logistic_loss, summary_stats
from src.sgd import sgd


#run the 30 tests for one sigma
def run_config(sigma, n, get_training_dataset, test_set, num_trials=NUM_TRIALS):
    losses = []
    errors = []

    for trial_id in range(1, num_trials + 1):
        #gets a fresh training stream to run sgd 
        training_dataset = get_training_dataset(n, sigma, trial_id)
        #w_hat  
        w_hat = sgd(training_dataset, n)

        losses.append(mean_logistic_loss(w_hat, test_set))
        errors.append(mean_classification_error(w_hat, test_set))

    loss_stats = summary_stats(losses)
    error_stats = summary_stats(errors)

    return {
        "sigma": sigma,
        "n": n,
        "losses": losses,
        "errors": errors,
        "loss_mean": loss_stats.mean,
        "loss_min": loss_stats.minimum,
        "loss_std": loss_stats.std,
        "estimated_excess_risk": loss_stats.mean - loss_stats.minimum,
        "error_mean": error_stats.mean,
        "error_std": error_stats.std,
    }

#run the all over all sigma
def run_experiments(
    get_training_dataset,
    get_fixed_test_set,
    sigmas=SIGMAS,
    training_sizes=TRAINING_SIZES,
    num_trials=NUM_TRIALS,
):
    results = []

    for sigma in sigmas:
        test_set = tuple(get_fixed_test_set(sigma))

        for n in training_sizes:
            results.append(
                run_config(
                    sigma=sigma,
                    n=n,
                    get_training_dataset=get_training_dataset,
                    test_set=test_set,
                    num_trials=num_trials,
                )
            )

    return results
