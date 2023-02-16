import os.path
import tempfile
import unittest

import numpy as np
import tensorflow as tf
from tensorflow import keras

import rlearn
from rlearn.trainer import tools
from rlearn.trainer.base import BaseTrainer


def get_default_ddpg_trainer():
    trainer = rlearn.trainer.DDPGTrainer()
    actor_encoder = keras.Sequential([
        keras.layers.InputLayer(2),
        keras.layers.Dense(32),
        keras.layers.ReLU(),
    ])
    critic_encoder = keras.Sequential([
        keras.layers.InputLayer(2),
        keras.layers.Dense(32),
        keras.layers.ReLU(),
    ])
    trainer.set_model_encoder(actor=actor_encoder, critic=critic_encoder, action_num=3)
    trainer.set_params(
        learning_rate=0.01,
        batch_size=32,
    )
    return trainer


class TrainerTest(unittest.TestCase):
    def test_get_trainer_map(self):
        d = {}
        rlearn.trainer.tools._set_trainer_map(BaseTrainer, d)
        for name in ["PPOContinueTrainer", "DuelingDQNTrainer", "DQNTrainer"]:
            self.assertIn(name, d)

    def test_build_from_conf(self):
        conf = rlearn.TrainConfig(
            trainer="DQNTrainer",
            batch_size=32,
            epochs=1000,
            action_transform=[0, 1],
            nets=[rlearn.NetConfig(
                input_shape=(4,),
                layers=[
                    rlearn.LayerConfig("dense", args={"units": 20}),
                    rlearn.LayerConfig("relu"),
                ]
            )],
            gamma=0.9,
            learning_rates=(0.01,),
            replay_buffer=rlearn.ReplayBufferConfig(500),
            replace_step=100,
            not_learn_epochs=5,
            epsilon_decay=0.01,
            min_epsilon=0.1,
        )
        trainer = rlearn.trainer.get_trainer_by_name(
            conf.trainer,
            log_dir=os.path.join(tempfile.gettempdir(), "test_dqn")
        )
        rlearn.trainer.set_config_to_trainer(conf, trainer)
        self.assertIn("q_", trainer.model.models)
        self.assertIn("q", trainer.model.models)

    def test_class_name(self):
        for k, v in rlearn.trainer.tools.get_all().items():
            self.assertEqual(k, v.name)

    def test_ddpg_add_encoder_manually(self):
        trainer = rlearn.trainer.DDPGTrainer(
            log_dir=os.path.join(tempfile.gettempdir(), "test_ddpg"))
        trainer.set_model_encoder(
            actor=keras.Sequential([
                keras.layers.InputLayer(2),
                keras.layers.Dense(16)
            ]),
            critic=keras.Sequential([
                keras.layers.InputLayer(2),
                keras.layers.Dense(32)
            ]),
            action_num=1
        )
        trainer.set_params(learning_rate=0.01)
        self.assertEqual((None, 2), trainer.model.models["actor"].input_shape)
        self.assertEqual((None, 1), trainer.model.models["actor"].output_shape)

    def test_ddpg_add_model(self):
        trainer = rlearn.trainer.DDPGTrainer()
        a = keras.Sequential([
            keras.layers.InputLayer(2),
            keras.layers.Dense(16),
            keras.layers.Dense(2)
        ])

        class C(keras.Model):
            def __init__(self):
                super().__init__()
                self.l1 = keras.layers.Dense(32)
                self.o = keras.layers.Dense(1)

            def call(self, x):
                # x is [s, a]
                o = tf.concat(x, axis=1)
                o = self.l1(o)
                return self.o(o)

        trainer.set_model(
            actor=a,
            critic=C(),
        )

        self.assertEqual((None, 2), trainer.model.models["actor"].input_shape)
        self.assertEqual((None, 2), trainer.model.models["actor"].output_shape)

        trainer.set_replay_buffer(8)
        trainer.set_params(learning_rate=0.01, batch_size=8)
        for _ in range(8):
            trainer.store_transition(np.random.random(2), np.random.random(2), np.random.random(), np.random.random(2))
        trainer.train_batch()

    def test_dqn_add_model(self):
        trainer = rlearn.trainer.DQNTrainer(
            log_dir=os.path.join(tempfile.gettempdir(), "test_dqn"))
        trainer.set_model(
            q=keras.Sequential([
                keras.layers.InputLayer(2),
                keras.layers.Dense(16),
                keras.layers.Dense(3)
            ])
        )
        self.assertEqual((None, 2), trainer.model.models["q"].input_shape)
        self.assertEqual((None, 3), trainer.model.models["q"].output_shape)
        pred = trainer.predict(np.zeros([2, ]))
        self.assertIsInstance(pred, int)

    def test_dqn(self):
        trainer = rlearn.trainer.DQNTrainer(
            log_dir=os.path.join(tempfile.gettempdir(), "test_dqn"))
        net = keras.Sequential([
            keras.layers.InputLayer(2),
            keras.layers.Dense(32),
            keras.layers.ReLU(),
            keras.layers.Dense(32),
            keras.layers.ReLU(),
        ])
        trainer.set_model_encoder(net, 3)
        trainer.set_replay_buffer(100)
        replace_ratio = 0.1
        trainer.set_params(
            learning_rate=[0.01, ],
            batch_size=32,
            replace_ratio=replace_ratio
        )
        self.assertEqual(1, trainer.epsilon)
        self.assertIsInstance(trainer.replay_buffer, rlearn.RandomReplayBuffer)
        v = trainer.model.models["q"].trainable_variables[0][0][0].numpy()
        v_ = trainer.model.models["q_"].trainable_variables[0][0][0].numpy()
        replaced = trainer.try_replace_params(source=trainer.model.models["q"],
                                              target=trainer.model.models["q_"])
        self.assertTrue(replaced)
        self.assertAlmostEqual(
            trainer.model.models["q_"].trainable_variables[0][0][0].numpy(),
            v_ * (1 - replace_ratio) + v * replace_ratio)
        self.assertIsInstance(trainer.predict(np.zeros([2, ])), int)

    def test_ppo_continuous(self):
        trainer = rlearn.PPOContinueTrainer()
        trainer.set_model_encoder(
            pi=keras.Sequential([
                keras.layers.InputLayer((2,)),
                keras.layers.Dense(10),
            ]),
            critic=keras.Sequential([
                keras.layers.InputLayer((2,)),
                keras.layers.Dense(10),
            ]),
            action_num=1
        )
        trainer.set_params(
            learning_rate=[0.01, 0.01],
            batch_size=32,
            min_epsilon=0.1,
            epsilon_decay=5e-5,
            gamma=0.9,
            replace_step=4,
        )
        action_transformer = rlearn.transformer.ContinuousAction([[0, 360]])

        self.assertEqual(1, trainer.epsilon)
        for _ in range(2):
            a = trainer.model.predict(np.random.random((2,)))
            a = action_transformer.transform(a).ravel()[0]
            self.assertTrue(0 <= a <= 360, msg=f"{a}")

    def test_ppo_discrete(self):
        trainer = rlearn.PPODiscreteTrainer()
        trainer.set_model_encoder(
            pi=keras.Sequential([
                keras.layers.InputLayer((2,)),
                keras.layers.Dense(10),
            ]),
            critic=keras.Sequential([
                keras.layers.InputLayer((2,)),
                keras.layers.Dense(10),
            ]),
            action_num=1
        )
        trainer.set_replay_buffer(1000)
        trainer.set_params(
            learning_rate=[0.001, 0.001],
            batch_size=32,
            min_epsilon=0.1,
            epsilon_decay=5e-5,
            gamma=0.9,
            replace_step=4,
        )

        for _ in range(2):
            a = trainer.model.predict(np.random.random((2,)))
            self.assertIsInstance(a, int, msg=f"{a}")

    def test_sac_continuous(self):
        trainer = rlearn.SACContinueTrainer()
        trainer.set_model_encoder(
            actor=keras.Sequential([
                keras.layers.InputLayer((2,)),
                keras.layers.Dense(10),
            ]),
            critic=keras.Sequential([
                keras.layers.InputLayer((2,)),
                keras.layers.Dense(10),
            ]),
            action_num=1
        )
        trainer.set_params(
            learning_rate=[0.01, 0.01],
            batch_size=32,
            min_epsilon=0.1,
            epsilon_decay=5e-5,
            gamma=0.9,
            replace_step=4,
        )
        action_transformer = rlearn.transformer.ContinuousAction([[0, 360]])

        self.assertEqual(1, trainer.epsilon)
        for _ in range(2):
            a = trainer.model.predict(np.random.random((2,)))
            a = action_transformer.transform(a).ravel()[0]
            self.assertTrue(0 <= a <= 360, msg=f"{a}")

    def test_sac_discrete(self):
        trainer = rlearn.SACDiscreteTrainer()
        trainer.set_model_encoder(
            actor=keras.Sequential([
                keras.layers.InputLayer((2,)),
                keras.layers.Dense(10),
            ]),
            critic=keras.Sequential([
                keras.layers.InputLayer((2,)),
                keras.layers.Dense(10),
            ]),
            action_num=1
        )
        trainer.set_replay_buffer(1000)
        trainer.set_params(
            learning_rate=[0.001, 0.001],
            batch_size=32,
            min_epsilon=0.1,
            epsilon_decay=5e-5,
            gamma=0.9,
            replace_step=4,
        )

        for _ in range(2):
            a = trainer.model.predict(np.random.random((2,)))
            self.assertIsInstance(a, int, msg=f"{a}")

    def test_dueling_dqn(self):
        trainer = rlearn.trainer.DuelingDQNTrainer()
        net = keras.Sequential([
            keras.layers.InputLayer(2),
            keras.layers.Dense(32),
            keras.layers.ReLU(),
            keras.layers.Dense(32),
            keras.layers.ReLU(),
        ])
        trainer.set_model_encoder(net, 3)
        replace_ratio = 0.1
        trainer.set_params(
            learning_rate=0.01,
            batch_size=32,
            replace_ratio=replace_ratio
        )
        self.assertIsInstance(trainer.predict(np.zeros([2, ])), int)
        self.assertEqual(rlearn.DuelingDQN.name, trainer.model.name)

    def test_ddpg(self):
        trainer = get_default_ddpg_trainer()
        pred = trainer.predict(np.zeros([2, ]))
        self.assertIsInstance(pred, np.ndarray)
        self.assertEqual(3, len(pred))
        self.assertEqual(rlearn.DDPG.name, trainer.model.name)

    def test_save_ckpt_model(self):
        trainer = get_default_ddpg_trainer()
        path = "tmp_model.zip"
        trainer.save_model_weights(path)
        self.assertTrue(os.path.isfile(path))
        trainer.load_model_weights(path)
        os.remove(path)

    def test_save_pb_model(self):
        trainer = get_default_ddpg_trainer()
        path = "tmp_model.zip"
        trainer.save_model(path)
        self.assertTrue(os.path.isfile(path))
        trainer.load_model(path)
        os.remove(path)

    def test_rnd(self):
        trainer = get_default_ddpg_trainer()
        trainer.add_rnd(target=keras.Sequential([
            keras.layers.InputLayer(2),
            keras.layers.ReLU(),
            keras.layers.Dense(10),
        ]))
        r = trainer.try_combine_int_ext_reward(1, np.random.random((2,)))
        self.assertIsInstance(r, float)
        r = trainer.try_combine_int_ext_reward([1, 2, 4], np.random.random((3, 2)))
        self.assertEqual((3,), r.shape)
        self.assertIsInstance(r, np.ndarray)

        r = trainer.try_combine_int_ext_reward(np.array([1, 2, 4]), np.random.random((3, 2)))
        self.assertEqual((3,), r.shape)
        self.assertIsInstance(r, np.ndarray)

    def test_save_pb_load_directly(self):
        trainer = rlearn.DQNTrainer()
        trainer.set_model_encoder(keras.Sequential([
            keras.layers.InputLayer(2),
            keras.layers.Dense(32),
        ]), action_num=3)
        path = "tmp_model0"
        trainer.save_model(path)
        self.assertTrue(os.path.exists(path + ".zip"))
        m = rlearn.load_model(path)
        self.assertFalse(os.path.exists(path))
        self.assertIsInstance(m.predict(np.random.random((2,))), int)
        os.remove(path + ".zip")

    def test_compute_flat_grads(self):
        trainer = rlearn.ActorCriticDiscreteTrainer()
        trainer.set_model_encoder(
            actor=keras.Sequential([
                keras.layers.InputLayer(2),
                keras.layers.Dense(3),
                keras.layers.ReLU(),
                keras.layers.Dense(5)
            ]),
            critic=keras.Sequential([
                keras.layers.InputLayer(2),
                keras.layers.Dense(3),
                keras.layers.ReLU(),
            ]),
            action_num=2
        )
        trainer.set_params(batch_size=3)
        trainer.set_replay_buffer()
        for _ in range(5):
            trainer.store_transition(
                s=np.random.random((2,)),
                a=np.random.randint(0, 2),
                r=np.random.random(),
                s_=np.random.random((2,)),
                done=False
            )
        grads = trainer.compute_flat_gradients()
        self.assertEqual(1, grads.ndim)
        self.assertEqual(sum(
            [w.size for w in trainer.model.models["actor"].get_weights()
             + trainer.model.models["critic"].get_weights()]), grads.size)

        trainer.apply_flat_gradients(grads)


class TrainerToolTest(unittest.TestCase):
    def test_parse_lr(self):
        l1, l2 = tools.parse_2_learning_rate(0.1)
        self.assertEqual(l1, l2)
        self.assertEqual(0.1, l1)

        l1, l2 = tools.parse_2_learning_rate([0.1, 0.2])
        self.assertEqual(0.1, l1)
        self.assertEqual(0.2, l2)

        l1, l2 = tools.parse_2_learning_rate([0.1, ])
        self.assertEqual(0.1, l1)
        self.assertEqual(0.1, l2)

        with self.assertRaises(ValueError) as cm:
            tools.parse_2_learning_rate([])

        self.assertEqual("the sequence length of the learning rate must greater than 1", str(cm.exception))

    def test_general_average_estimation(self):
        model = keras.Sequential([
            keras.layers.InputLayer(2),
            keras.layers.Dense(1),
        ])
        returns, adv = tools.general_average_estimation(
            value_model=model,
            batch_s=np.random.random((3, 2)),
            batch_done=[False, False, False],
            batch_r=np.random.random((3,)),
            s_=np.random.random((2,)),
            gamma=0.9,
            lam=0.9
        )
        self.assertEqual((3,), returns.shape)
        self.assertEqual((3,), adv.shape)

    def test_discounted_reward_adv(self):
        model = keras.Sequential([
            keras.layers.InputLayer(2),
            keras.layers.Dense(1),
        ])
        bs = np.random.random((3, 2))
        returns = tools.discounted_reward(
            value_model=model,
            batch_s=bs,
            batch_done=[False, False, False],
            batch_r=np.random.random((3,)),
            s_=np.random.random((2,)),
            gamma=0.9,
        )
        self.assertEqual((3,), returns.shape)

        adv = tools.discounted_adv(value_model=model, batch_s=bs, reward=returns)
        self.assertEqual((3,), adv.shape)

    def test_reshape_flat_gradients(self):
        m1 = keras.Sequential([
            keras.layers.InputLayer(2),
            keras.layers.Dense(1),
        ])
        m2 = keras.Sequential([
            keras.layers.InputLayer(2),
            keras.layers.Dense(3),
            keras.layers.Dense(1),
        ])
        m3 = keras.Sequential([
            keras.layers.InputLayer(2),
            keras.layers.Dense(4),
        ])
        s = sum([np.prod(p.shape) for p in m1.trainable_variables])
        s += sum([np.prod(p.shape) for p in m2.trainable_variables])
        s += sum([np.prod(p.shape) for p in m3.trainable_variables])
        reshaped = tools.reshape_flat_gradients(
            {"a": [m1, m2], "b": [m3]},
            gradients=np.random.random(s))
        self.assertEqual(6, len(reshaped["a"]))
        self.assertEqual(2, len(reshaped["b"]))
