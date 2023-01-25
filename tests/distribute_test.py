import json
import multiprocessing
import os
import shutil
import time
import unittest

import grpc
import numpy as np
from tensorflow import keras

import rlearn
from rlearn import distribute
from rlearn.distribute import buffer_pb2, buffer_pb2_grpc, actor_pb2_grpc, actor_pb2
from rlearn.distribute import tools
from tests.gym_wrapper_test import CartPoleSmoothReward


class BufferTest(unittest.TestCase):
    server = None
    stub = None
    port = tools.get_available_port()

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = distribute.buffer._start_server(
            port=cls.port,
            max_size=100,
            buf="RandomReplayBuffer",
            debug=True,
        )
        channel = grpc.insecure_channel(f'localhost:{cls.port}')
        cls.stub = buffer_pb2_grpc.ReplayBufferStub(channel=channel)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.stop(None)

    def test_ready(self):
        resp = self.stub.ServiceReady(buffer_pb2.ServiceReadyReq())
        self.assertTrue(resp.ready)

    def test_put(self):
        req = buffer_pb2.UploadDataReq()
        req.data.values[:] = [1, 2, 3, 4, 5, 6, 7, 8]
        req.data.attributes = json.dumps([
            {"name": "s", "shape": [2, 2]},
            {"name": "a", "shape": [2, 1]},
            {"name": "r", "shape": [2, 1]}]
        )
        resp = self.stub.UploadData(req)
        self.assertTrue(resp.done)
        self.assertEqual("", resp.err)

    def test_download(self):
        self.test_put()
        resp = self.stub.DownloadData(buffer_pb2.DownloadDataReq())
        _, batch = tools.unpack_transitions(resp)
        self.assertEqual(2, batch["s"].shape[1])
        self.assertEqual(1, batch["a"].shape[1])
        self.assertEqual(1, batch["r"].shape[1])
        self.assertTrue("s_" not in batch)


class ActorProcessTest(unittest.TestCase):
    model_pb_path = os.path.join(os.path.dirname(__file__), os.pardir, "tmp", "test_distribute_dqn_pb.zip")
    model_ckpt_path = os.path.join(os.path.dirname(__file__), os.pardir, "tmp", "test_distribute_dqn.zip")

    def setUp(self) -> None:
        model = rlearn.zoo.DQNSmall(4, 2)
        model.save(self.model_pb_path)
        model.save_weights(self.model_ckpt_path)

    def test_rl_env(self):
        env = CartPoleSmoothReward(seed=1)
        s = env.reset()
        self.assertIsInstance(s, np.ndarray)
        self.assertEqual((4,), s.shape)
        s_, r, done = env.step(0)
        self.assertIsInstance(s_, np.ndarray)
        self.assertEqual((4,), s_.shape)
        self.assertIsInstance(r, float)
        self.assertIsInstance(done, bool)

    def test_ep_step_generator(self):
        env = CartPoleSmoothReward()
        actor_p = distribute.actor.ActorProcess(
            local_buffer_size=500,
            env=env,
            remote_buffer_address=None,
            action_transformer=None,
        )
        actor_p.init_params(
            "DQN", self.model_pb_path, init_version=0, request_id="dqn_train", max_episode=2, max_episode_step=20)
        g = tools.get_count_generator(-1)
        for i in range(10):
            step = next(g)
            self.assertEqual(i, step)

        g = tools.get_count_generator(3)
        for i in range(10):
            if i < 3:
                step = next(g)
                self.assertEqual(i, step)
            else:
                with self.assertRaises(StopIteration):
                    next(g)
                break

    def test_actor_process(self):
        env = CartPoleSmoothReward()
        actor = distribute.actor.ActorProcess(
            local_buffer_size=500,
            env=env,
            remote_buffer_address=None,
            action_transformer=None,
        )
        actor.init_params(
            "DQNTrainer", self.model_pb_path, init_version=0,
            request_id="dqn_train", max_episode=2, max_episode_step=5)
        actor.start()
        actor.join()
        self.assertEqual(1, actor.ns.episode_num)
        self.assertLess(actor.ns.episode_step_num, 5)

    def test_actor_process_in_process(self):
        buf_port = tools.get_available_port()
        buf_server = distribute.buffer._start_server(
            port=buf_port,
            max_size=100,
            buf="RandomReplayBuffer",
            debug=True,
        )
        buf_address = f'localhost:{buf_port}'
        channel = grpc.insecure_channel(buf_address)
        buf_stub = buffer_pb2_grpc.ReplayBufferStub(channel=channel)

        init_version = 0
        resp = buf_stub.LearnerSetVersion(buffer_pb2.LearnerSetVersionReq(version=init_version, requestId="bl"))
        self.assertEqual("bl", resp.requestId)
        self.assertTrue(resp.done)
        self.assertEqual("", resp.err)

        env = CartPoleSmoothReward()
        actor_p = distribute.actor.ActorProcess(
            local_buffer_size=10,
            env=env,
            remote_buffer_address=buf_address,
            action_transformer=None,
        )
        actor_p.init_params(
            "DQNTrainer",
            self.model_pb_path,
            init_version=init_version,
            request_id="dqn_train",
            max_episode=2,
            max_episode_step=20)
        actor_p.start()
        actor_p.ns.new_model_path = self.model_ckpt_path
        actor_p.join()

        self.assertEqual(1, actor_p.ns.episode_num)
        self.assertLess(actor_p.ns.episode_step_num, 20)
        resp = buf_stub.DownloadData(buffer_pb2.DownloadDataReq(maxSize=10, requestId="xx"))
        self.assertEqual("xx", resp.requestId)
        batch_size, batch = tools.unpack_transitions(resp)
        self.assertGreater(batch_size, 0)
        buf_server.stop(None)

        self.assertEqual(4, batch["s"].shape[1])
        self.assertEqual(1, batch["a"].ndim)
        self.assertEqual(1, batch["r"].ndim)
        self.assertEqual(4, batch["s_"].shape[1])
        for i in batch.values():
            self.assertGreaterEqual(i.shape[0], 10)


class ActorServiceTest(unittest.TestCase):
    buf_server = None
    buf_stub = None
    actor_server = None
    actor_stub = None
    model_pb_path = os.path.join(os.path.dirname(__file__), os.pardir, "tmp", "test_distribute_dqn_pb.zip")
    model_ckpt_path = os.path.join(os.path.dirname(__file__), os.pardir, "tmp", "test_distribute_dqn.zip")
    ps = []

    @classmethod
    def setUpClass(cls) -> None:
        buf_port = tools.get_available_port()
        buf_address = f'localhost:{buf_port}'
        cls.buf_server = distribute.buffer._start_server(
            port=buf_port,
            max_size=100,
            buf="RandomReplayBuffer",
            debug=True,
        )
        buf_channel = grpc.insecure_channel(buf_address)
        cls.buf_stub = buffer_pb2_grpc.ReplayBufferStub(channel=buf_channel)

        actor_port = tools.get_available_port()
        cls.actor_server = distribute.actor._start_server(
            port=actor_port,
            remote_buffer_address=buf_address,
            local_buffer_size=3,
            env=CartPoleSmoothReward(),
            action_transformer=None,
            debug=True,
        )
        actor_address = f'localhost:{actor_port}'
        actor_channel = grpc.insecure_channel(actor_address)
        cls.actor_stub = actor_pb2_grpc.ActorStub(channel=actor_channel)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.actor_stub.Terminate(actor_pb2.TerminateReq(requestId="tt"))
        cls.actor_server.stop(None)
        cls.buf_server.stop(None)

    def setUp(self) -> None:
        model = rlearn.zoo.DQNSmall(4, 2)
        model.save(self.model_pb_path)
        model.save_weights(self.model_ckpt_path)

    def test_ready(self):
        resp = self.actor_stub.ServiceReady(actor_pb2.ServiceReadyReq(requestId="xx"))
        self.assertTrue(resp.ready)
        self.assertEqual("xx", resp.requestId)

    def test_train(self):
        version = 0
        self.buf_stub.LearnerSetModelType(buffer_pb2.LearnerSetModelTypeReq(isOnPolicy=False))
        resp = self.buf_stub.LearnerSetVersion(buffer_pb2.LearnerSetVersionReq(version=version, requestId="bl"))
        self.assertEqual("bl", resp.requestId)
        self.assertTrue(resp.done)
        self.assertEqual("", resp.err)

        resp = self.actor_stub.Start(tools.read_pb_iterfile(
            self.model_pb_path,
            trainer_type="DQNTrainer",
            max_episode=0,
            max_episode_step=0,
            version=version,
            request_id="start"
        ))
        self.assertEqual("start", resp.requestId)
        self.assertTrue(resp.done)
        self.assertEqual("", resp.err)

        time.sleep(0.5)
        next_version = 1
        self.buf_stub.LearnerSetVersion(buffer_pb2.LearnerSetVersionReq(version=next_version, requestId="bl"))
        resp = self.actor_stub.ReplicateModel(tools.read_weights_iterfile(
            self.model_ckpt_path,
            version=next_version,
            request_id="replicate"
        ))
        self.assertEqual("replicate", resp.requestId)
        self.assertTrue(resp.done)
        self.assertEqual("", resp.err)

        resp = self.buf_stub.DownloadData(buffer_pb2.DownloadDataReq(maxSize=3, requestId="bufDownload"))
        batch_size, batch = tools.unpack_transitions(resp)
        self.assertEqual(3, batch_size)
        self.assertIsInstance(batch["s"], np.ndarray)
        self.assertIsInstance(batch["a"], np.ndarray)
        self.assertIsInstance(batch["r"], np.ndarray)
        self.assertEqual((3,), batch["r"].shape)
        self.assertIsInstance(batch["s_"], np.ndarray)
        self.assertEqual((3, 4), batch["s"].shape)
        self.assertEqual((3, 4), batch["s_"].shape)
        self.assertEqual("bufDownload", resp.requestId)


class LearnerTest(unittest.TestCase):
    buf_address = None
    actors_address = []
    ps = []
    result_dir = os.path.join(os.path.dirname(__file__), os.pardir, "tmp", "dist_learner_test")

    @classmethod
    def setUpClass(cls) -> None:
        buf_port = tools.get_available_port()
        cls.buf_address = buf_address = f'localhost:{buf_port}'
        p = multiprocessing.Process(target=distribute.start_replay_buffer_server, kwargs=dict(
            port=buf_port,
            max_size=1000,
            buf="RandomReplayBuffer",
            # debug=True,
        ))
        p.start()
        cls.ps.append(p)

        for _ in range(2):
            actor_port = tools.get_available_port()
            p = multiprocessing.Process(target=distribute.start_actor_server, kwargs=dict(
                port=actor_port,
                remote_buffer_address=buf_address,
                local_buffer_size=10,
                env=CartPoleSmoothReward(),
                action_transformer=None,
                # debug=True,
            ))
            p.start()
            cls.ps.append(p)
            actor_address = f'localhost:{actor_port}'
            cls.actors_address.append(actor_address)

    @classmethod
    def tearDownClass(cls) -> None:
        [p.terminate() for p in cls.ps]
        shutil.rmtree(cls.result_dir, ignore_errors=True)

    def test_run(self):
        trainer = rlearn.trainer.DQNTrainer()
        trainer.set_replay_buffer()
        trainer.set_model_encoder(
            q=keras.Sequential([
                keras.layers.InputLayer(4),
                keras.layers.Dense(10),
            ]),
            action_num=2
        )
        trainer.set_params(
            learning_rate=0.01,
            batch_size=32,
            replace_step=15,
        )
        learner = distribute.Learner(
            trainer=trainer,
            remote_buffer_address=self.buf_address,
            remote_actors_address=self.actors_address,
            result_dir=self.result_dir,
            debug=True,
        )
        learner.run(epoch=3, epoch_step=None)
