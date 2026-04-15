# CSE 5523 Project: Stochastic Gradient Descent for Logistic Regression

This repository contains a course project for CSE 5523 on single-pass projected SGD for logistic regression. The code follows the project setup with feature dimension `d = 4`, parameter dimension `d + 1 = 5`, logistic loss, projection onto the unit ball, and evaluation over the required `(sigma, n)` experiment grid.


**Project Settings**

- `sigma in {0.2, 0.4}`
- `n in {50, 100, 500, 1000}`
- `N = 400` test examples
- `30` trials per `(sigma, n)`
- parameter set `C = {w in R^5 : ||w|| <= 1}`
- feature set `X = {x in R^4 : ||x|| <= 1}`
- step size `alpha = M / (rho * sqrt(n + 1))`
- `M = 2`
- `rho = sqrt(2)`

**Code Flow**

1. Run [scripts/run_all.py](/Users/allenthomas/Code/CSE5523/scripts/run_all.py).
2. It calls [src/experiment.py](/Users/allenthomas/Code/CSE5523/src/experiment.py).
3. The experiment driver gets training datasets and fixed test sets from [src/data.py](/Users/allenthomas/Code/CSE5523/src/data.py).
4. Each trial calls [src/sgd.py](/Users/allenthomas/Code/CSE5523/src/sgd.py).
5. SGD uses:
   [src/loss.py](/Users/allenthomas/Code/CSE5523/src/loss.py) for logistic loss and gradient,
   [src/projection.py](/Users/allenthomas/Code/CSE5523/src/projection.py) for projection,
   and [src/config.py](/Users/allenthomas/Code/CSE5523/src/config.py) for constants and step size.
6. The experiment driver evaluates the output predictor with [src/metrics.py](/Users/allenthomas/Code/CSE5523/src/metrics.py).
7. Results are written to `results/experiment_results.json`.
8. Run [scripts/make_plots.py](/Users/allenthomas/Code/CSE5523/scripts/make_plots.py) to generate the PNG plots in `results/plots`.

**Repository Layout**

- [src/config.py](/Users/allenthomas/Code/CSE5523/src/config.py): constants and step size
- [src/data.py](/Users/allenthomas/Code/CSE5523/src/data.py): data generation and fixed test sets
- [src/experiment.py](/Users/allenthomas/Code/CSE5523/src/experiment.py): trial loop and experiment summaries
- [src/loss.py](/Users/allenthomas/Code/CSE5523/src/loss.py): logistic loss and stochastic gradient
- [src/metrics.py](/Users/allenthomas/Code/CSE5523/src/metrics.py): evaluation metrics and summary statistics
- [src/plots.py](/Users/allenthomas/Code/CSE5523/src/plots.py): error-bar plotting helpers
- [src/projection.py](/Users/allenthomas/Code/CSE5523/src/projection.py): projection onto the unit ball
- [src/sgd.py](/Users/allenthomas/Code/CSE5523/src/sgd.py): single-pass projected SGD
- [scripts/run_all.py](/Users/allenthomas/Code/CSE5523/scripts/run_all.py): main experiment runner
- [scripts/make_plots.py](/Users/allenthomas/Code/CSE5523/scripts/make_plots.py): plot generator


**How To Run**

Run all experiments:

```bash
python3 scripts/run_all.py
```

This writes:

- `results/experiment_results.json`

Generate plots from the saved JSON:

```bash
python3 scripts/make_plots.py
```

This writes:

- `results/plots/sigma_02_excess_risk.png`
- `results/plots/sigma_02_classification_error.png`
- `results/plots/sigma_04_excess_risk.png`
- `results/plots/sigma_04_classification_error.png`

**Output Format**

Each row in `results/experiment_results.json` contains:

- `sigma`
- `n`
- `losses`
- `errors`
- `loss_mean`
- `loss_min`
- `loss_std`
- `estimated_excess_risk`
- `error_mean`
- `error_std`

`losses` and `errors` each contain the `30` trial-level estimates for that `(sigma, n)` setting.

**Current Results**

These are the results currently stored in [results/experiment_results.json](/Users/allenthomas/Code/CSE5523/results/experiment_results.json).

| sigma | n | N | trials | Logistic Loss Mean | Logistic Loss STD | Logistic Loss Min | Excess Risk | Classification Error Mean | Classification Error STD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | 50 | 400 | 30 | 0.5292348105 | 0.0072771074 | 0.5165897388 | 0.0126450717 | 0.0400833333 | 0.0336803809 |
| 0.2 | 100 | 400 | 30 | 0.5125990072 | 0.0055922006 | 0.5052952925 | 0.0073037147 | 0.0203333333 | 0.0164409719 |
| 0.2 | 500 | 400 | 30 | 0.4930356633 | 0.0016991576 | 0.4893413603 | 0.0036943030 | 0.0119166667 | 0.0021098315 |
| 0.2 | 1000 | 400 | 30 | 0.4888564458 | 0.0013492712 | 0.4869625973 | 0.0018938485 | 0.0132500000 | 0.0020564938 |
| 0.4 | 50 | 400 | 30 | 0.5592126052 | 0.0116530882 | 0.5417615367 | 0.0174510685 | 0.1499166667 | 0.0462262702 |
| 0.4 | 100 | 400 | 30 | 0.5424647334 | 0.0061948001 | 0.5322336830 | 0.0102310505 | 0.1190833333 | 0.0157120354 |
| 0.4 | 500 | 400 | 30 | 0.5241715814 | 0.0021772086 | 0.5209309781 | 0.0032406033 | 0.1065000000 | 0.0040620192 |
| 0.4 | 1000 | 400 | 30 | 0.5193830088 | 0.0014601724 | 0.5171783267 | 0.0022046821 | 0.1050833333 | 0.0039519686 |

**Notes**

- The learner is single-pass SGD, not an ERM-style loop over a stored dataset with resampling.
- The data generator acts as the distribution side of the project. Once learning starts, the SGD code only consumes realized examples.
- The report, template, and appendix text are separate from this README. This file is only meant to document the codebase and how to run it.
