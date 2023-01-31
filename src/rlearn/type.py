import typing as tp

import numpy as np

Action = tp.TypeVar("Action", int, float, np.ndarray)
LearningRate = tp.TypeVar("LearningRate", tp.Sequence[float], float)
BatchReward = tp.TypeVar('BatchReward', tp.List[float], np.ndarray)
