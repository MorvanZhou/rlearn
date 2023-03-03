import inspect
import os
import shutil
import unittest

import numpy as np
from tensorflow import keras

import rlearn


class SupervisedLearnTest(unittest.TestCase):
    def setUp(self) -> None:
        self.save_dir = os.path.join(os.path.dirname(__file__), os.pardir, "tmp", "super_test")
        os.makedirs(self.save_dir, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.save_dir)

    def test_all_model(self):
        x = np.random.random((128, 4))
        y = np.random.randint(0, 2, size=(128,))
        for k, trainer_class in rlearn.trainer.tools.get_all().items():
            trainer = trainer_class()
            with self.assertRaises(ValueError) as cm:
                rlearn.supervised.fit(
                    trainer=trainer,
                    x=x,
                    y=y,
                    epoch=2,
                    batch_size=32,
                    shuffle=True,
                    learning_rate=1e-3,
                    model_save_dir=os.path.join(self.save_dir, k),
                    verbose=1,
                )
                self.assertEqual(
                    "trainer's model has not been set, please add encoder or model to this trainer",
                    str(cm),
                )
            models = [
                keras.Sequential([
                    keras.layers.InputLayer(4),
                    keras.layers.Dense(2)
                ]),
                keras.Sequential([
                    keras.layers.InputLayer(4),
                    keras.layers.Dense(2)
                ]),
            ]
            sp = inspect.signature(trainer.set_model_encoder).parameters
            trainer.set_model_encoder(
                *(models if len(sp) > 2 else models[0:1]),
                action_num=2 if trainer.model.is_discrete_action else 1,
            )
            model_dir = os.path.join(self.save_dir, k)
            rlearn.supervised.fit(
                trainer=trainer,
                x=x,
                y=y,
                epoch=2,
                batch_size=32,
                shuffle=True,
                learning_rate=1e-3,
                model_save_dir=model_dir,
                # verbose=1,
            )

            rlearn.supervised.set_actor_weights(trainer, os.path.join(model_dir, "ep-1.zip"))
