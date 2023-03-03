import numpy as np

import rlearn.trainer.base


def fit(
        trainer: rlearn.trainer.base.BaseTrainer,
        x: np.ndarray,
        y: np.ndarray,
        epoch: int,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        shuffle: bool = True,
        model_save_dir: str = None,
        verbose: int = 0,
):
    trainer.train_supervised(
        x=x,
        y=y,
        epoch=epoch,
        batch_size=batch_size,
        learning_rate=learning_rate,
        shuffle=shuffle,
        save_dir=model_save_dir,
        verbose=verbose,
    )


def set_actor_weights(
        trainer: rlearn.trainer.base.BaseTrainer,
        path: str,
):
    trainer.model.load_actor_weights(path)
