import json
import multiprocessing
import os
import shutil
import time
import typing as tp
import unittest
import zlib

import grpc
import numpy as np
from tensorflow import keras

import rlearn
from rlearn import distributed
from rlearn.distributed import tools
from rlearn.distributed.experience import buffer_pb2, buffer_pb2_grpc, actor_pb2_grpc, actor_pb2
from tests.test_gym_wrapper import CartPoleSmoothReward


class BufferTest(unittest.TestCase):
    server = None
    stub = None
    port = tools.get_available_port()

    @classmethod
    def setUpClass(cls) -> None:
        cls.server, _ = distributed.experience.buffer._start_server(
            port=cls.port,
            debug=True,
        )
        channel = grpc.insecure_channel(f'localhost:{cls.port}')
        cls.stub = buffer_pb2_grpc.ReplayBufferStub(channel=channel)
        cls.stub.InitBuf(buffer_pb2.InitBufReq(isOnPolicy=False, bufferType="RandomReplayBuffer", bufferSize=100))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.stop(None)

    def test_ready(self):
        resp = self.stub.ServiceReady(buffer_pb2.ServiceReadyReq())
        self.assertTrue(resp.ready)

    def test_put(self):
        def iter_req():
            req = buffer_pb2.UploadDataReq(
                meta=buffer_pb2.UploadDataMeta(
                    version=1,
                    attributes=json.dumps([
                        {"name": "s", "shape": [2, 2]},
                        {"name": "a", "shape": [2, 1]},
                        {"name": "r", "shape": [2, 1]}]
                    ),
                    requestId="xxx",
                )
            )
            yield req
            v = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
            bv = v.tobytes()
            cbv = zlib.compress(bv)
            for i in range(0, len(cbv), 1024):
                req = buffer_pb2.UploadDataReq(
                    chunkData=cbv[i: i + 1024]
                )
                yield req

        resp = self.stub.UploadData(iter_req())
        self.assertTrue(resp.done)
        self.assertEqual("", resp.err)

    def test_download(self):
        self.test_put()
        resp_iter = self.stub.DownloadData(buffer_pb2.DownloadDataReq())
        _, batch, err, _ = tools.unpack_downloaded_transitions(resp_iter=resp_iter)
        self.assertEqual("", err)
        self.assertEqual(2, batch["s"].shape[1])
        self.assertEqual(1, batch["a"].shape[1])
        self.assertEqual(1, batch["r"].shape[1])
        self.assertTrue("s_" not in batch)


class ExperienceActorProcessTest(unittest.TestCase):
    model_pb_path = os.path.join(os.path.dirname(__file__), os.pardir, "tmp", "test_distribute_dqn_pb.zip")
    recv_conn, send_conn = multiprocessing.Pipe(False)

    def setUp(self) -> None:
        model = rlearn.zoo.smallDQN(4, 2)
        model.save(self.model_pb_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.recv_conn.close()
        cls.send_conn.close()
        os.remove(cls.model_pb_path)

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
        actor_p = distributed.experience.actor.ActorProcess(
            weights_conn=self.recv_conn,
            env=env,
            remote_buffer_address=None,
        )
        actor_p.init_params(
            "DQN",
            model_pb_path=self.model_pb_path,
            init_version=0,
            buffer_size=500,
            request_id="dqn_train", max_episode=2, max_episode_step=20)
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
        actor = distributed.experience.actor.ActorProcess(
            weights_conn=self.recv_conn,
            env=env,
            remote_buffer_address=None,
        )
        actor.init_params(
            "DQNTrainer",
            model_pb_path=self.model_pb_path,
            init_version=0,
            buffer_size=500,
            request_id="dqn_train", max_episode=2, max_episode_step=5)
        actor.start()
        actor.join()
        self.assertEqual(1, actor.ns.episode_num)
        self.assertLess(actor.ns.episode_step_num, 5)

    def test_actor_process_in_process(self):
        buf_port = tools.get_available_port()
        buf_server, _ = distributed.experience.buffer._start_server(
            port=buf_port,
            debug=True,
        )
        buf_address = f'localhost:{buf_port}'
        channel = grpc.insecure_channel(buf_address)
        buf_stub = buffer_pb2_grpc.ReplayBufferStub(channel=channel)

        resp = buf_stub.InitBuf(buffer_pb2.InitBufReq(
            bufferType="RandomReplayBuffer",
            bufferSize=100,
            isOnPolicy=True,
        ))
        self.assertTrue(resp.done)

        init_version = 0
        resp = buf_stub.LearnerSetVersion(buffer_pb2.LearnerSetVersionReq(version=init_version, requestId="bl"))
        self.assertEqual("bl", resp.requestId)
        self.assertTrue(resp.done)
        self.assertEqual("", resp.err)

        env = CartPoleSmoothReward()
        actor_p = distributed.experience.actor.ActorProcess(
            weights_conn=self.recv_conn,
            env=env,
            remote_buffer_address=buf_address,
        )
        actor_p.init_params(
            "DQNTrainer",
            self.model_pb_path,
            init_version=init_version,
            buffer_size=10,
            request_id="dqn_train",
            max_episode=2,
            max_episode_step=20)
        actor_p.start()
        actor_p.join()

        self.assertEqual(1, actor_p.ns.episode_num)
        self.assertLess(actor_p.ns.episode_step_num, 20)
        resp_iter = buf_stub.DownloadData(buffer_pb2.DownloadDataReq(maxSize=10, requestId="xx"))
        batch_size, batch, err, req_id = tools.unpack_downloaded_transitions(resp_iter=resp_iter)
        self.assertEqual("xx", req_id)
        self.assertEqual("", err)
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
    ps = []

    @classmethod
    def setUpClass(cls) -> None:
        buf_port = tools.get_available_port()
        buf_address = f'localhost:{buf_port}'
        cls.buf_server, _ = distributed.experience.buffer._start_server(
            port=buf_port,
            debug=True,
        )
        buf_channel = grpc.insecure_channel(buf_address)
        cls.buf_stub = buffer_pb2_grpc.ReplayBufferStub(channel=buf_channel)

        actor_port = tools.get_available_port()
        cls.actor_server, _, _ = distributed.experience.actor._start_server(
            port=actor_port,
            remote_buffer_address=buf_address,
            env=CartPoleSmoothReward(),
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
        os.remove(cls.model_pb_path)

    def setUp(self) -> None:
        model = rlearn.zoo.smallDQN(4, 2)
        model.save(self.model_pb_path)
        self.model_weighs = model.get_flat_weights()

    def test_ready(self):
        resp = self.actor_stub.ServiceReady(actor_pb2.ServiceReadyReq(requestId="xx"))
        self.assertTrue(resp.ready)
        self.assertEqual("xx", resp.requestId)

    def test_train(self):
        version = 0
        self.buf_stub.InitBuf(buffer_pb2.InitBufReq(
            isOnPolicy=False,
            bufferSize=100,
            bufferType="RandomReplayBuffer",
        ))
        resp = self.buf_stub.LearnerSetVersion(buffer_pb2.LearnerSetVersionReq(version=version, requestId="bl"))
        self.assertEqual("bl", resp.requestId)
        self.assertTrue(resp.done)
        self.assertEqual("", resp.err)

        resp = self.actor_stub.Start(tools.read_pb_iterfile(
            self.model_pb_path,
            trainer_type="DQNTrainer",
            buffer_size=10,
            buffer_type="RandomReplayBuffer",
            max_episode=0,
            max_episode_step=0,
            action_transform=[0, 1],
            version=version,
            request_id="start"
        ))
        self.assertEqual("start", resp.requestId)
        self.assertTrue(resp.done)
        self.assertEqual("", resp.err)

        time.sleep(0.1)
        next_version = 1
        self.buf_stub.LearnerSetVersion(buffer_pb2.LearnerSetVersionReq(version=next_version, requestId="bl"))
        resp = self.actor_stub.ReplicateModel(tools.get_iter_values(
            msg_handler=actor_pb2.ReplicateModelReq,
            values=self.model_weighs,
            version=next_version,
            request_id="replicate"
        ))
        self.assertEqual("replicate", resp.requestId)
        self.assertTrue(resp.done)
        self.assertEqual("", resp.err)

        resp_iter = self.buf_stub.DownloadData(buffer_pb2.DownloadDataReq(maxSize=3, requestId="bufDownload"))
        batch_size, batch, err, req_id = tools.unpack_downloaded_transitions(resp_iter=resp_iter)
        self.assertEqual("", err)
        self.assertEqual(3, batch_size)
        self.assertIsInstance(batch["s"], np.ndarray)
        self.assertIsInstance(batch["a"], np.ndarray)
        self.assertIsInstance(batch["r"], np.ndarray)
        self.assertEqual((3,), batch["r"].shape)
        self.assertIsInstance(batch["s_"], np.ndarray)
        self.assertEqual((3, 4), batch["s"].shape)
        self.assertEqual((3, 4), batch["s_"].shape)
        self.assertEqual("bufDownload", req_id)


class LearnerTest(unittest.TestCase):
    buf_address = None
    actors_address = []
    ps: tp.List[multiprocessing.Process] = []
    result_dir = os.path.join(os.path.dirname(__file__), os.pardir, "tmp", "dist_learner_test")

    @classmethod
    def setUpClass(cls) -> None:
        buf_port = tools.get_available_port()
        cls.buf_address = buf_address = f'localhost:{buf_port}'
        p = multiprocessing.Process(target=distributed.experience.start_replay_buffer_server, kwargs=dict(
            port=buf_port,
            debug=True,
        ))
        p.start()
        cls.ps.append(p)

        for _ in range(2):
            actor_port = tools.get_available_port()
            p = multiprocessing.Process(target=distributed.experience.start_actor_server, kwargs=dict(
                port=actor_port,
                remote_buffer_address=buf_address,
                env=CartPoleSmoothReward(),
                debug=True,
            ))
            p.start()
            cls.ps.append(p)
            actor_address = f'localhost:{actor_port}'
            cls.actors_address.append(actor_address)

    @classmethod
    def tearDownClass(cls) -> None:
        [p.join() for p in cls.ps]
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
        trainer.set_action_transformer(rlearn.transformer.DiscreteAction([0, 1]))
        learner = distributed.experience.Learner(
            trainer=trainer,
            remote_buffer_address=self.buf_address,
            actors_address=self.actors_address,
            remote_buffer_size=1000,
            remote_buffer_type="RandomReplayBuffer",
            actor_buffer_size=10,
            save_dir=self.result_dir,
            debug=True,
        )
        learner.run(max_train_time=1, max_ep_step=-1)
