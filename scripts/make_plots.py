import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plots import plot_classification_error, plot_excess_risk


def load_results(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    import matplotlib.pyplot as plt

    input_path = Path("results/experiment_results.json")
    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(input_path)
    sigmas = sorted({result["sigma"] for result in results})

    for sigma in sigmas:
        fig, _ = plot_excess_risk(
            results,
            sigma=sigma,
            output_path=output_dir / f"sigma_{str(sigma).replace('.', '')}_excess_risk.png",
        )
        plt.close(fig)

        fig, _ = plot_classification_error(
            results,
            sigma=sigma,
            output_path=output_dir / f"sigma_{str(sigma).replace('.', '')}_classification_error.png",
        )
        plt.close(fig)

    print(f"Wrote plots for sigma values {sigmas} to {output_dir}")


if __name__ == "__main__":
    main()
