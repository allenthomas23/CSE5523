import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import get_fixed_test_set, get_training_dataset
from src.experiment import run_experiments


def main():
    output_path = Path("results/experiment_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    #run the experiments
    results = run_experiments(
        get_training_dataset=get_training_dataset,
        get_fixed_test_set=get_fixed_test_set,
    )
    #write result json
    output_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    print("Wrote " + str(len(results)) + " experiment summaries to " + str(output_path))


if __name__ == "__main__":
    main()
