"""
Example showing how to tune multiple objectives at once of an artificial function.
"""
import logging
from pathlib import Path

import numpy as np

from syne_tune.backend import LocalBackend
from syne_tune.optimizer.baselines import NSGA2
from syne_tune import Tuner, StoppingCriterion
from syne_tune.config_space import uniform
from syne_tune.config_space import (
    lograndint, uniform, loguniform, choice,
)

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    np.random.seed(0)

    max_steps = 27
    n_workers = 1
    config_space = {
    'lr': loguniform(1e-4, 0.8),
    'edge1': choice(["0","1","2","3","4","5","6","7","8"]),
    'edge2': choice(["0","1","2","3","4","5","6","7","8"]),
    'edge3': choice(["0","1","2","3","4","5","6","7","8"]),
    'head' : choice(["CosFace","ArcFace","MagFace"]),
    'optimizer': choice(["Adam","SGD","AdamW"]),
    
    }
    entry_point = ("src/search/dpn_objective_synetune.py")
    mode = "min"

    np.random.seed(0)
    scheduler = NSGA2(
        max_t=max_steps,
        mode=["min", "min"],
        metric=["rev_acc", "disparity"],
        config_space=config_space,
    )
    trial_backend = LocalBackend(entry_point=str(entry_point),num_gpus_per_trial=8)

    stop_criterion = StoppingCriterion(max_wallclock_time=100000000000000000000000000000000000000000000000000000000000)
    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=n_workers,
        sleep_time=0.5,

    )
    tuner.run()