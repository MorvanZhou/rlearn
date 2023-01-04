# import json
# import unittest
#
# import grpc
#
# from rllearn.distributed import buffer_pb2
# from rllearn.distributed import replaybuf
# from rllearn.distributed.buffer_pb2_grpc import ReplayBufferStub
# from rllearn.matrix import unpack_transitions
# from rllearn.tools import get_available_port
#
#
# class MemoryTest(unittest.TestCase):
#     game_svr = None
#     server = None
#     stub = None
#     port = get_available_port()
#
#     @classmethod
#     def setUpClass(cls) -> None:
#         _grpc_server, _thread_service = replaybuf.start_server(
#             port=cls.port,
#             buf_name="RandomReplayBuffer",
#             max_size=100,
#             debug=True,
#         )
#         cls.game_svr = _thread_service
#         cls.server = _grpc_server
#         channel = grpc.insecure_channel(f'127.0.0.1:{cls.port}')
#         cls.stub = ReplayBufferStub(channel=channel)
#
#     @classmethod
#     def tearDownClass(cls) -> None:
#         cls.server.stop(None)
#
#     def test_ready(self):
#         resp = self.stub.ServiceReady(buffer_pb2.ServiceReadyReq())
#         self.assertTrue(resp.ready)
#
#     def test_put(self):
#         req = buffer_pb2.UploadDataReq()
#         req.data.values[:] = [1, 2, 3, 4, 5, 6, 7, 8]
#         req.data.attributes = json.dumps({
#             "s_shape": [2, 2],
#             "a_shape": [2, 1],
#             "r_shape": [2, 1],
#             "has_next_state": False,
#         })
#         resp = self.stub.UploadData(req)
#         self.assertTrue(resp.done)
#         self.assertEqual("", resp.err)
#
#     def test_download(self):
#         self.test_put()
#         resp = self.stub.DownloadData(buffer_pb2.DownloadDataReq())
#         s, a, r, s_ = unpack_transitions(resp)
#         self.assertEqual(2, s.shape[1])
#         self.assertEqual(1, a.shape[1])
#         self.assertEqual(1, r.shape[1])
#         self.assertIsNone(s_)
