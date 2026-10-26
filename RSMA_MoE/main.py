from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import ExperimentConfig
from run_requested_sensitivity_figures import (
    METHOD_ORDER,
    SWEEP_VALUES,
    run_requested_figures,
)


def main() -> None:
    config = ExperimentConfig()
    run_requested_figures(
        runs=100,
        max_attempts_per_point=None,
        method_keys=METHOD_ORDER,
        sweep_names=tuple(SWEEP_VALUES),
        quick=False,
        dag_seed=config.random_seed,
        network_seed=config.network_random_seed,
    )


if __name__ == "__main__":
    main()
