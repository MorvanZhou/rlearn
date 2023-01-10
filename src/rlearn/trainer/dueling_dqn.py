from rlearn.model.dueling_dqn import DuelingDQN
from rlearn.trainer.dqn import DQNTrainer


class DuelingDQNTrainer(DQNTrainer):
    name = __qualname__

    def __init__(
            self,
            log_dir: str = None
    ):
        super().__init__(log_dir)
        self.model = DuelingDQN(training=True)
