import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import get_fixed_test_set, get_training_dataset
from src.experiment import run_project_experiments


def main():
    output_path = Path("results/experiment_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        results = run_project_experiments(
            get_training_dataset=get_training_dataset,
            get_fixed_test_set=get_fixed_test_set,
        )
    except NotImplementedError as exc:
        raise SystemExit(
            "Data-generation helpers are not implemented yet. "
            "Fill src.data.get_training_dataset and src.data.get_fixed_test_set first."
        ) from exc

    output_path.write_text(
        json.dumps([result.__dict__ for result in results], indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(results)} experiment summaries to {output_path}")


if __name__ == "__main__":
    main()
