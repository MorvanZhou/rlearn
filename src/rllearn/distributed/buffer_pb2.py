# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tipealgs/ml/rl/distributed/buffer.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()

DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\'tipealgs/ml/rl/distributed/buffer.proto\x12\x0creplayBuffer\"\x11\n\x0fServiceReadyReq\"!\n\x10ServiceReadyResp\x12\r\n\x05ready\x18\x01 \x01(\x08\"*\n\x04\x44\x61ta\x12\x0e\n\x06values\x18\x01 \x03(\x02\x12\x12\n\nattributes\x18\x02 \x01(\t\"1\n\rUploadDataReq\x12 \n\x04\x64\x61ta\x18\x01 \x01(\x0b\x32\x12.replayBuffer.Data\"+\n\x0eUploadDataResp\x12\x0c\n\x04\x64one\x18\x01 \x01(\x08\x12\x0b\n\x03\x65rr\x18\x02 \x01(\t\"\"\n\x0f\x44ownloadDataReq\x12\x0f\n\x07maxSize\x18\x01 \x01(\x05\"A\n\x10\x44ownloadDataResp\x12 \n\x04\x64\x61ta\x18\x01 \x01(\x0b\x32\x12.replayBuffer.Data\x12\x0b\n\x03\x65rr\x18\x02 \x01(\t2\xfb\x01\n\x0cReplayBuffer\x12O\n\x0cServiceReady\x12\x1d.replayBuffer.ServiceReadyReq\x1a\x1e.replayBuffer.ServiceReadyResp\"\x00\x12I\n\nUploadData\x12\x1b.replayBuffer.UploadDataReq\x1a\x1c.replayBuffer.UploadDataResp\"\x00\x12O\n\x0c\x44ownloadData\x12\x1d.replayBuffer.DownloadDataReq\x1a\x1e.replayBuffer.DownloadDataResp\"\x00\x42\x31\n\x14io.grpc.examples.envB\x11ReplayBufferProtoP\x01\xa2\x02\x03HLWb\x06proto3')

_SERVICEREADYREQ = DESCRIPTOR.message_types_by_name['ServiceReadyReq']
_SERVICEREADYRESP = DESCRIPTOR.message_types_by_name['ServiceReadyResp']
_DATA = DESCRIPTOR.message_types_by_name['Data']
_UPLOADDATAREQ = DESCRIPTOR.message_types_by_name['UploadDataReq']
_UPLOADDATARESP = DESCRIPTOR.message_types_by_name['UploadDataResp']
_DOWNLOADDATAREQ = DESCRIPTOR.message_types_by_name['DownloadDataReq']
_DOWNLOADDATARESP = DESCRIPTOR.message_types_by_name['DownloadDataResp']
ServiceReadyReq = _reflection.GeneratedProtocolMessageType('ServiceReadyReq', (_message.Message,), {
    'DESCRIPTOR': _SERVICEREADYREQ,
    '__module__': 'tipealgs.ml.rl.distributed.buffer_pb2'
    # @@protoc_insertion_point(class_scope:replayBuffer.ServiceReadyReq)
})
_sym_db.RegisterMessage(ServiceReadyReq)

ServiceReadyResp = _reflection.GeneratedProtocolMessageType('ServiceReadyResp', (_message.Message,), {
    'DESCRIPTOR': _SERVICEREADYRESP,
    '__module__': 'tipealgs.ml.rl.distributed.buffer_pb2'
    # @@protoc_insertion_point(class_scope:replayBuffer.ServiceReadyResp)
})
_sym_db.RegisterMessage(ServiceReadyResp)

Data = _reflection.GeneratedProtocolMessageType('Data', (_message.Message,), {
    'DESCRIPTOR': _DATA,
    '__module__': 'tipealgs.ml.rl.distributed.buffer_pb2'
    # @@protoc_insertion_point(class_scope:replayBuffer.Data)
})
_sym_db.RegisterMessage(Data)

UploadDataReq = _reflection.GeneratedProtocolMessageType('UploadDataReq', (_message.Message,), {
    'DESCRIPTOR': _UPLOADDATAREQ,
    '__module__': 'tipealgs.ml.rl.distributed.buffer_pb2'
    # @@protoc_insertion_point(class_scope:replayBuffer.UploadDataReq)
})
_sym_db.RegisterMessage(UploadDataReq)

UploadDataResp = _reflection.GeneratedProtocolMessageType('UploadDataResp', (_message.Message,), {
    'DESCRIPTOR': _UPLOADDATARESP,
    '__module__': 'tipealgs.ml.rl.distributed.buffer_pb2'
    # @@protoc_insertion_point(class_scope:replayBuffer.UploadDataResp)
})
_sym_db.RegisterMessage(UploadDataResp)

DownloadDataReq = _reflection.GeneratedProtocolMessageType('DownloadDataReq', (_message.Message,), {
    'DESCRIPTOR': _DOWNLOADDATAREQ,
    '__module__': 'tipealgs.ml.rl.distributed.buffer_pb2'
    # @@protoc_insertion_point(class_scope:replayBuffer.DownloadDataReq)
})
_sym_db.RegisterMessage(DownloadDataReq)

DownloadDataResp = _reflection.GeneratedProtocolMessageType('DownloadDataResp', (_message.Message,), {
    'DESCRIPTOR': _DOWNLOADDATARESP,
    '__module__': 'tipealgs.ml.rl.distributed.buffer_pb2'
    # @@protoc_insertion_point(class_scope:replayBuffer.DownloadDataResp)
})
_sym_db.RegisterMessage(DownloadDataResp)

_REPLAYBUFFER = DESCRIPTOR.services_by_name['ReplayBuffer']
if _descriptor._USE_C_DESCRIPTORS == False:
    DESCRIPTOR._options = None
    DESCRIPTOR._serialized_options = b'\n\024io.grpc.examples.envB\021ReplayBufferProtoP\001\242\002\003HLW'
    _SERVICEREADYREQ._serialized_start = 57
    _SERVICEREADYREQ._serialized_end = 74
    _SERVICEREADYRESP._serialized_start = 76
    _SERVICEREADYRESP._serialized_end = 109
    _DATA._serialized_start = 111
    _DATA._serialized_end = 153
    _UPLOADDATAREQ._serialized_start = 155
    _UPLOADDATAREQ._serialized_end = 204
    _UPLOADDATARESP._serialized_start = 206
    _UPLOADDATARESP._serialized_end = 249
    _DOWNLOADDATAREQ._serialized_start = 251
    _DOWNLOADDATAREQ._serialized_end = 285
    _DOWNLOADDATARESP._serialized_start = 287
    _DOWNLOADDATARESP._serialized_end = 352
    _REPLAYBUFFER._serialized_start = 355
    _REPLAYBUFFER._serialized_end = 606
# @@protoc_insertion_point(module_scope)
