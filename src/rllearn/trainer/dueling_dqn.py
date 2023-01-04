import typing as tp

from rllearn.model.dueling_dqn import DuelingDQN
from rllearn.trainer.dqn import DQNTrainer


class DuelingDQNTrainer(DQNTrainer):
    name = __qualname__

    def __init__(
            self,
            learning_rates: tp.Sequence[float],
            log_dir: str = None
    ):
        super().__init__(learning_rates, log_dir)
        self.model = DuelingDQN(training=True)

