import multiprocessing
import typing as tp
import unittest

import grpc
import numpy as np
from tensorflow import keras

import rlearn
from rlearn import distributed
from rlearn.distributed import tools, logger
from rlearn.distributed.gradient import param_pb2, param_pb2_grpc, worker
from rlearn.distributed.gradient.param_server import _start_server
from tests.test_gym_wrapper import CartPoleSmoothReward


class GradientParamTest(unittest.TestCase):

    def test_param_service(self):
        trainer = rlearn.DQNTrainer()
        trainer.set_model_encoder(q=keras.Sequential([
            keras.layers.InputLayer(3),
            keras.layers.Dense(5),
        ]), action_num=2)
        trainer.set_params(
            learning_rate=0.01,
            batch_size=32,
            gamma=0.8,
        )
        trainer.set_action_transformer(rlearn.transformer.DiscreteAction([0, 1]))
        port = tools.get_available_port()
        server, stop_event = _start_server(
            port=port,
            trainer=trainer,
            sync_step=5,
            worker_buffer_size=1000,
            worker_buffer_type="RandomReplayBuffer",
            max_ep_step=-1,
            debug=True
        )
        address = f'localhost:{port}'
        channel = grpc.insecure_channel(address)
        stub = param_pb2_grpc.ParamsStub(channel=channel)
        lg = logger.get_logger("worker")

        w_trainer, max_episode_step, sync_step = worker.init(lg, stub)
        self.assertEqual(-1, max_episode_step)
        self.assertEqual(5, sync_step)
        self.assertAlmostEqual(0.8, w_trainer.gamma)
        self.assertEqual(32, w_trainer.batch_size)

        s = np.random.random(3)
        _a = w_trainer.predict(s)
        a = w_trainer.map_action(_a)
        self.assertIsInstance(a, int)
        s_, r, done = np.random.random(3), 1, False

        [w_trainer.store_transition(s=s, a=_a, r=r, s_=s_, done=done) for _ in range(5)]

        stop = worker.sync(logger=lg, stub=stub, trainer=w_trainer)
        self.assertFalse(stop)

        resp = stub.Terminate(param_pb2.TerminateReq(requestId="t"))
        self.assertEqual("t", resp.requestId)
        self.assertEqual("", resp.err)
        self.assertEqual(True, resp.done)

        stop = worker.sync(logger=lg, stub=stub, trainer=w_trainer)
        self.assertTrue(stop)

        stop_event.wait()
        server.stop(None)
        server.wait_for_termination()


class ParamServerTest(unittest.TestCase):
    def test_run(self):
        context = multiprocessing.get_context('spawn')
        ps: tp.List[context.Process] = []

        trainer = rlearn.DQNTrainer()
        trainer.set_model_encoder(q=keras.Sequential([
            keras.layers.InputLayer(4),
            keras.layers.Dense(20),
            keras.layers.ReLU(),
        ]), action_num=2)
        trainer.set_params(
            learning_rate=0.01,
            batch_size=32,
            gamma=0.8,
            replace_step=100,
        )
        trainer.set_action_transformer(rlearn.transformer.DiscreteAction([0, 1]))

        port = tools.get_available_port()

        for _ in range(2):
            p = context.Process(target=distributed.gradient.worker.run, kwargs=dict(
                env=CartPoleSmoothReward(),
                params_server_address=f"localhost:{port}",
                debug=True,
            ))
            p.start()
            ps.append(p)

        distributed.gradient.start_param_server(
            port=port,
            trainer=trainer,
            sync_step=5,
            worker_buffer_size=1000,
            worker_buffer_type="RandomReplayBuffer",
            max_ep_step=-1,
            max_train_time=2,
            debug=True
        )
        [p.join() for p in ps]
        [p.terminate() for p in ps]
