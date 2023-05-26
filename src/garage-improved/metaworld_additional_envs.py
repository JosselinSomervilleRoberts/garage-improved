import metaworld
import metaworld.envs.mujoco.env_dict as _env_dict

from collections import OrderedDict
from typing import Optional

from metaworld.envs.mujoco.sawyer_xyz.v2 import (
    SawyerPickPlaceEnvV2,
    SawyerPushEnvV2,
    SawyerReachEnvV2,
    SawyerDoorEnvV2,
    SawyerDrawerOpenEnvV2,
    SawyerDrawerCloseEnvV2,
    SawyerButtonPressTopdownEnvV2,
    SawyerPegInsertionSideEnvV2,
    SawyerWindowOpenEnvV2,
    SawyerWindowCloseEnvV2,
)

# =========== Custom MT3 (3 first tasks of MT10) =========== #

MT3_V2 = OrderedDict(
    (('reach-v2', SawyerReachEnvV2),
     ('push-v2', SawyerPushEnvV2),
     ('pick-place-v2', SawyerPickPlaceEnvV2) )
)

MT3_V2_ARGS_KWARGS = {
    key: dict(args=[],
              kwargs={'task_id': list(_env_dict.ALL_V2_ENVIRONMENTS.keys()).index(key)})
    for key, _ in MT3_V2.items()
}

class MT3(metaworld.Benchmark):
    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = MT3_V2
        self._test_classes = OrderedDict()
        train_kwargs = MT3_V2_ARGS_KWARGS
        self._train_tasks = metaworld._make_tasks(self._train_classes, train_kwargs,
                                        metaworld._MT_OVERRIDE,
                                        seed=seed)
        self._test_tasks = []

# ========================================================= #



# =========== Custom MTFlexible (N first tasks of MT10) =========== #

# Ordered by increasing difficulty (according to the paper)
MTFLEXIBLE_V2 = OrderedDict(
    (('reach-v2', SawyerReachEnvV2),
     ('drawer-close-v2', SawyerDrawerCloseEnvV2),
     ('drawer-open-v2', SawyerDrawerOpenEnvV2),
     ('window-open-v2', SawyerWindowOpenEnvV2),
     ('button-press-topdown-v2', SawyerButtonPressTopdownEnvV2),
     ('window-close-v2', SawyerWindowCloseEnvV2),
     ('door-open-v2', SawyerDoorEnvV2),
     ('push-v2', SawyerPushEnvV2),
     ('peg-insert-side-v2', SawyerPegInsertionSideEnvV2),
     ('pick-place-v2', SawyerPickPlaceEnvV2)),)
     
class MTFlexible(metaworld.Benchmark):
    def __init__(self, n: int, increasing_difficulty: bool = True, task_name: Optional[str] = None, seed=None):
        super().__init__()
        assert n >= 1, "n must be >= 1"
        assert n <= 10, "n must be <= 10, as it currently uses MT10"
        # Keep the first n tasks of MTN_V2
        if n == 1:
            assert task_name is not None, "task_name must be provided if n == 1"
            self._train_classes = OrderedDict([(task_name, _env_dict.ALL_V2_ENVIRONMENTS[task_name])])
        else:
            self._train_classes = OrderedDict(list(MTFLEXIBLE_V2.items())[:n]) if increasing_difficulty else OrderedDict(list(MTFLEXIBLE_V2.items())[-n:])
        self._test_classes = OrderedDict()
        train_kwargs = {key: dict(args=[],
                                    kwargs={'task_id': list(_env_dict.ALL_V2_ENVIRONMENTS.keys()).index(key)})
                        for key, _ in self._train_classes.items()}
        self._train_tasks = metaworld._make_tasks(self._train_classes, train_kwargs,
                                        metaworld._MT_OVERRIDE,
                                        seed=seed)
        self._test_tasks = [] 

# ========================================================= #
