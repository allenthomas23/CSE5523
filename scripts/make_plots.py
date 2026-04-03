import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment import ExperimentSettingResult
from src.plots import plot_classification_error_vs_n, plot_excess_risk_vs_n


def load_results(path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [
        ExperimentSettingResult(
            sigma=row["sigma"],
            n=row["n"],
            trial_losses=tuple(row["trial_losses"]),
            trial_errors=tuple(row["trial_errors"]),
            loss_mean=row["loss_mean"],
            loss_min=row["loss_min"],
            loss_std=row["loss_std"],
            estimated_excess_risk=row["estimated_excess_risk"],
            error_mean=row["error_mean"],
            error_std=row["error_std"],
        )
        for row in rows
    ]


def main():
    import matplotlib.pyplot as plt

    input_path = Path("results/experiment_results.json")
    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(input_path)
    sigmas = sorted({result.sigma for result in results})

    for sigma in sigmas:
        fig, _ = plot_excess_risk_vs_n(
            results,
            sigma=sigma,
            output_path=output_dir / f"sigma_{str(sigma).replace('.', '')}_excess_risk.png",
        )
        plt.close(fig)

        fig, _ = plot_classification_error_vs_n(
            results,
            sigma=sigma,
            output_path=output_dir / f"sigma_{str(sigma).replace('.', '')}_classification_error.png",
        )
        plt.close(fig)

    print(f"Wrote plots for sigma values {sigmas} to {output_dir}")


if __name__ == "__main__":
    main()
