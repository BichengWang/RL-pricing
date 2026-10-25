"""Exercise the authors' run_DRL entry point with five A2C learning steps."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from stable_baselines.common.vec_env import DummyVecEnv


def main() -> None:
    checkout = Path.cwd()
    sys.path.insert(0, str(checkout))
    entrypoint = checkout / "run_DRL.py"
    spec = importlib.util.spec_from_file_location("authors_run_drl", entrypoint)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load authors' entry point: {entrypoint}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def smoke_ensemble(df, unique_trade_date, rebalance_window, validation_window):
        del unique_trade_date, rebalance_window, validation_window
        training = module.data_split(df, start=20090000, end=20090112)
        env = DummyVecEnv([lambda: module.StockEnvTrain(training)])
        model = module.train_A2C(env, model_name="r1_l1_probe", timesteps=5)
        assert model.num_timesteps == 5, model.num_timesteps
        print("PASS: authors' run_DRL.py entry point built and trained A2C for 5 steps")

    module.run_ensemble_strategy = smoke_ensemble
    module.run_model()


if __name__ == "__main__":
    main()
