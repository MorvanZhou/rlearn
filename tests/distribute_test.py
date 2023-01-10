import json
import multiprocessing
import os
import shutil
import time
import unittest

import grpc
import gym
import numpy as np
from tensorflow import keras

import rlearn
from rlearn import distribute
from rlearn.distribute import buffer_pb2, buffer_pb2_grpc, actor_pb2_grpc, actor_pb2
from rlearn.distribute import tools


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
        channel = grpc.insecure_channel(f'127.0.0.1:{cls.port}')
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
        req.data.attributes = json.dumps({
            "s_shape": [2, 2],
            "a_shape": [2, 1],
            "r_shape": [2, 1],
            "has_next_state": False,
        })
        resp = self.stub.UploadData(req)
        self.assertTrue(resp.done)
        self.assertEqual("", resp.err)

    def test_download(self):
        self.test_put()
        resp = self.stub.DownloadData(buffer_pb2.DownloadDataReq())
        s, a, r, s_ = tools.unpack_transitions(resp)
        self.assertEqual(2, s.shape[1])
        self.assertEqual(1, a.shape[1])
        self.assertEqual(1, r.shape[1])
        self.assertIsNone(s_)


class CartPole(distribute.actor.RLEnv):
    def __init__(self, seed=None):
        self.env = gym.make('CartPole-v1', new_step_api=True)
        if seed is not None:
            self.env.reset(seed=seed)

    def reset(self):
        s = self.env.reset(return_info=False)
        return s

    def step(self, a):
        s_, _, done, _, _ = self.env.step(a)
        x, _, theta, _ = s_
        r1 = (self.env.x_threshold - abs(x)) / self.env.x_threshold - 0.8
        r2 = (self.env.theta_threshold_radians - abs(theta)) / self.env.theta_threshold_radians - 0.5
        r = r1 + r2
        return s_, r, done


class ActorProcessTest(unittest.TestCase):
    model_pb_path = os.path.join(os.path.dirname(__file__), os.pardir, "tmp", "test_distribute_dqn_pb.zip")
    model_ckpt_path = os.path.join(os.path.dirname(__file__), os.pardir, "tmp", "test_distribute_dqn.zip")

    def setUp(self) -> None:
        model = rlearn.zoo.DQNSmall(4, 2)
        model.save(self.model_pb_path)
        model.save_weights(self.model_ckpt_path)

    def test_rl_env(self):
        env = CartPole(seed=1)
        s = env.reset()
        self.assertIsInstance(s, np.ndarray)
        self.assertEqual((4,), s.shape)
        s_, r, done = env.step(0)
        self.assertIsInstance(s_, np.ndarray)
        self.assertEqual((4,), s_.shape)
        self.assertIsInstance(r, float)
        self.assertIsInstance(done, bool)

    def test_ep_step_generator(self):
        buffer = rlearn.RandomReplayBuffer(500)
        env = CartPole()
        actor_p = distribute.actor.ActorProcess(
            buffer=buffer,
            env=env,
            remote_buffer_address="",
            action_transformer=None,
        )
        actor_p.init_params(
            "DQN", self.model_pb_path, init_version="v0", request_id="dqn_train", max_episode=2, max_episode_step=20)
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
        buffer = rlearn.RandomReplayBuffer(500)
        env = CartPole()
        actor = distribute.actor.ActorProcess(
            buffer=buffer,
            env=env,
            remote_buffer_address="",
            action_transformer=None,
        )
        actor.init_params(
            "DQN", self.model_pb_path, init_version="v0", request_id="dqn_train", max_episode=2, max_episode_step=5)
        actor.start()
        actor.join()
        self.assertEqual(1, actor.ns.episode_num)
        self.assertLess(actor.ns.episode_step_num, 5)
        self.assertLess(buffer.current_loading_point, 5 * 2)

    def test_actor_process_in_process(self):
        buf_port = tools.get_available_port()
        buf_server = distribute.buffer._start_server(
            port=buf_port,
            max_size=100,
            buf="RandomReplayBuffer",
            debug=True,
        )
        buf_address = f'127.0.0.1:{buf_port}'
        channel = grpc.insecure_channel(buf_address)
        buf_stub = buffer_pb2_grpc.ReplayBufferStub(channel=channel)

        init_version = "v0"
        resp = buf_stub.LearnerSetVersion(buffer_pb2.LearnerSetVersionReq(version=init_version, requestId="bl"))
        self.assertEqual("bl", resp.requestId)
        self.assertTrue(resp.done)
        self.assertEqual("", resp.err)

        buffer = rlearn.RandomReplayBuffer(10)
        env = CartPole()
        actor_p = distribute.actor.ActorProcess(
            buffer=buffer,
            env=env,
            remote_buffer_address=buf_address,
            action_transformer=None,
        )
        actor_p.init_params(
            "DQN",
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
        s, a, r, s_ = tools.unpack_transitions(resp)
        buf_server.stop(None)

        self.assertEqual(4, s.shape[1])
        self.assertEqual(1, a.shape[1])
        self.assertEqual(1, r.shape[1])
        self.assertEqual(4, s_.shape[1])
        for i in [s, a, r, s_]:
            self.assertGreaterEqual(i.shape[0], buffer.max_size)


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
        buf_address = f'127.0.0.1:{buf_port}'
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
            max_local_buf_size=3,
            env=CartPole(),
            action_transformer=None,
            debug=True,
        )
        actor_address = f'127.0.0.1:{actor_port}'
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
        version = "v0"
        resp = self.buf_stub.LearnerSetVersion(buffer_pb2.LearnerSetVersionReq(version=version, requestId="bl"))
        self.assertEqual("bl", resp.requestId)
        self.assertTrue(resp.done)
        self.assertEqual("", resp.err)

        resp = self.actor_stub.Start(tools.read_pb_iterfile(
            self.model_pb_path,
            model_type="DQN",
            max_episode=0,
            max_episode_step=0,
            version=version,
            request_id="start"
        ))
        self.assertEqual("start", resp.requestId)
        self.assertTrue(resp.done)
        self.assertEqual("", resp.err)

        time.sleep(0.3)
        next_version = "v1"
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
        s, a, r, s_ = tools.unpack_transitions(resp)
        self.assertIsInstance(s, np.ndarray)
        self.assertIsInstance(a, np.ndarray)
        self.assertIsInstance(r, np.ndarray)
        self.assertEqual((3, 1), r.shape)
        self.assertIsInstance(s_, np.ndarray)
        self.assertEqual((3, 4), s.shape)
        self.assertEqual((3, 4), s_.shape)
        self.assertEqual("bufDownload", resp.requestId)
        for data in tools.unpack_transitions(resp):
            self.assertEqual(3, len(data))


class LearnerTest(unittest.TestCase):
    buf_address = None
    actors_address = []
    ps = []
    result_dir = os.path.join(os.path.dirname(__file__), os.pardir, "tmp", "dist_learner_test")

    @classmethod
    def setUpClass(cls) -> None:
        buf_port = tools.get_available_port()
        cls.buf_address = buf_address = f'127.0.0.1:{buf_port}'
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
                max_local_buf_size=10,
                env=CartPole(),
                action_transformer=None,
                # debug=True,
            ))
            p.start()
            cls.ps.append(p)
            actor_address = f'127.0.0.1:{actor_port}'
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
