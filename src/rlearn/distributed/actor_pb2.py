# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: rlearn/distributed/actor.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()

DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\x1erlearn/distributed/actor.proto\x12\x05\x61\x63tor\"$\n\x0fServiceReadyReq\x12\x11\n\trequestId\x18\x01 \x01(\t\"4\n\x10ServiceReadyResp\x12\r\n\x05ready\x18\x01 \x01(\x08\x12\x11\n\trequestId\x18\x02 \x01(\t\"\xc3\x01\n\tStartMeta\x12\x10\n\x08\x66ilename\x18\x01 \x01(\t\x12\x13\n\x0btrainerType\x18\x02 \x01(\t\x12\x12\n\nbufferType\x18\x03 \x01(\t\x12\x12\n\nbufferSize\x18\x04 \x01(\x04\x12\x12\n\nmaxEpisode\x18\x05 \x01(\x03\x12\x16\n\x0emaxEpisodeStep\x18\x06 \x01(\x03\x12\x17\n\x0f\x61\x63tionTransform\x18\x07 \x01(\t\x12\x0f\n\x07version\x18\x08 \x01(\x04\x12\x11\n\trequestId\x18\t \x01(\t\"L\n\x08StartReq\x12 \n\x04meta\x18\x01 \x01(\x0b\x32\x10.actor.StartMetaH\x00\x12\x13\n\tchunkData\x18\x02 \x01(\x0cH\x00\x42\t\n\x07request\"9\n\tStartResp\x12\x0c\n\x04\x64one\x18\x01 \x01(\x08\x12\x0b\n\x03\x65rr\x18\x02 \x01(\t\x12\x11\n\trequestId\x18\x03 \x01(\t\"/\n\tModelMeta\x12\x0f\n\x07version\x18\x01 \x01(\x04\x12\x11\n\trequestId\x18\x02 \x01(\t\"$\n\x0fGetGradientsReq\x12\x11\n\trequestId\x18\x01 \x01(\t\"T\n\x10GetGradientsResp\x12 \n\x04meta\x18\x01 \x01(\x0b\x32\x10.actor.ModelMetaH\x00\x12\x13\n\tchunkData\x18\x02 \x01(\x0cH\x00\x42\t\n\x07request\"U\n\x11ReplicateModelReq\x12 \n\x04meta\x18\x01 \x01(\x0b\x32\x10.actor.ModelMetaH\x00\x12\x13\n\tchunkData\x18\x02 \x01(\x0cH\x00\x42\t\n\x07request\"B\n\x12ReplicateModelResp\x12\x0c\n\x04\x64one\x18\x01 \x01(\x08\x12\x0b\n\x03\x65rr\x18\x02 \x01(\t\x12\x11\n\trequestId\x18\x03 \x01(\t\"!\n\x0cTerminateReq\x12\x11\n\trequestId\x18\x01 \x01(\t\"=\n\rTerminateResp\x12\x0c\n\x04\x64one\x18\x01 \x01(\x08\x12\x0b\n\x03\x65rr\x18\x02 \x01(\t\x12\x11\n\trequestId\x18\x03 \x01(\t2\xc4\x02\n\x05\x41\x63tor\x12\x41\n\x0cServiceReady\x12\x16.actor.ServiceReadyReq\x1a\x17.actor.ServiceReadyResp\"\x00\x12.\n\x05Start\x12\x0f.actor.StartReq\x1a\x10.actor.StartResp\"\x00(\x01\x12\x43\n\x0cGetGradients\x12\x16.actor.GetGradientsReq\x1a\x17.actor.GetGradientsResp\"\x00\x30\x01\x12I\n\x0eReplicateModel\x12\x18.actor.ReplicateModelReq\x1a\x19.actor.ReplicateModelResp\"\x00(\x01\x12\x38\n\tTerminate\x12\x13.actor.TerminateReq\x1a\x14.actor.TerminateResp\"\x00\x42*\n\x14io.grpc.examples.envB\nActorProtoP\x01\xa2\x02\x03HLWb\x06proto3')

_SERVICEREADYREQ = DESCRIPTOR.message_types_by_name['ServiceReadyReq']
_SERVICEREADYRESP = DESCRIPTOR.message_types_by_name['ServiceReadyResp']
_STARTMETA = DESCRIPTOR.message_types_by_name['StartMeta']
_STARTREQ = DESCRIPTOR.message_types_by_name['StartReq']
_STARTRESP = DESCRIPTOR.message_types_by_name['StartResp']
_MODELMETA = DESCRIPTOR.message_types_by_name['ModelMeta']
_GETGRADIENTSREQ = DESCRIPTOR.message_types_by_name['GetGradientsReq']
_GETGRADIENTSRESP = DESCRIPTOR.message_types_by_name['GetGradientsResp']
_REPLICATEMODELREQ = DESCRIPTOR.message_types_by_name['ReplicateModelReq']
_REPLICATEMODELRESP = DESCRIPTOR.message_types_by_name['ReplicateModelResp']
_TERMINATEREQ = DESCRIPTOR.message_types_by_name['TerminateReq']
_TERMINATERESP = DESCRIPTOR.message_types_by_name['TerminateResp']
ServiceReadyReq = _reflection.GeneratedProtocolMessageType('ServiceReadyReq', (_message.Message,), {
    'DESCRIPTOR': _SERVICEREADYREQ,
    '__module__': 'rlearn.distributed.actor_pb2'
    # @@protoc_insertion_point(class_scope:actor.ServiceReadyReq)
})
_sym_db.RegisterMessage(ServiceReadyReq)

ServiceReadyResp = _reflection.GeneratedProtocolMessageType('ServiceReadyResp', (_message.Message,), {
    'DESCRIPTOR': _SERVICEREADYRESP,
    '__module__': 'rlearn.distributed.actor_pb2'
    # @@protoc_insertion_point(class_scope:actor.ServiceReadyResp)
})
_sym_db.RegisterMessage(ServiceReadyResp)

StartMeta = _reflection.GeneratedProtocolMessageType('StartMeta', (_message.Message,), {
    'DESCRIPTOR': _STARTMETA,
    '__module__': 'rlearn.distributed.actor_pb2'
    # @@protoc_insertion_point(class_scope:actor.StartMeta)
})
_sym_db.RegisterMessage(StartMeta)

StartReq = _reflection.GeneratedProtocolMessageType('StartReq', (_message.Message,), {
    'DESCRIPTOR': _STARTREQ,
    '__module__': 'rlearn.distributed.actor_pb2'
    # @@protoc_insertion_point(class_scope:actor.StartReq)
})
_sym_db.RegisterMessage(StartReq)

StartResp = _reflection.GeneratedProtocolMessageType('StartResp', (_message.Message,), {
    'DESCRIPTOR': _STARTRESP,
    '__module__': 'rlearn.distributed.actor_pb2'
    # @@protoc_insertion_point(class_scope:actor.StartResp)
})
_sym_db.RegisterMessage(StartResp)

ModelMeta = _reflection.GeneratedProtocolMessageType('ModelMeta', (_message.Message,), {
    'DESCRIPTOR': _MODELMETA,
    '__module__': 'rlearn.distributed.actor_pb2'
    # @@protoc_insertion_point(class_scope:actor.ModelMeta)
})
_sym_db.RegisterMessage(ModelMeta)

GetGradientsReq = _reflection.GeneratedProtocolMessageType('GetGradientsReq', (_message.Message,), {
    'DESCRIPTOR': _GETGRADIENTSREQ,
    '__module__': 'rlearn.distributed.actor_pb2'
    # @@protoc_insertion_point(class_scope:actor.GetGradientsReq)
})
_sym_db.RegisterMessage(GetGradientsReq)

GetGradientsResp = _reflection.GeneratedProtocolMessageType('GetGradientsResp', (_message.Message,), {
    'DESCRIPTOR': _GETGRADIENTSRESP,
    '__module__': 'rlearn.distributed.actor_pb2'
    # @@protoc_insertion_point(class_scope:actor.GetGradientsResp)
})
_sym_db.RegisterMessage(GetGradientsResp)

ReplicateModelReq = _reflection.GeneratedProtocolMessageType('ReplicateModelReq', (_message.Message,), {
    'DESCRIPTOR': _REPLICATEMODELREQ,
    '__module__': 'rlearn.distributed.actor_pb2'
    # @@protoc_insertion_point(class_scope:actor.ReplicateModelReq)
})
_sym_db.RegisterMessage(ReplicateModelReq)

ReplicateModelResp = _reflection.GeneratedProtocolMessageType('ReplicateModelResp', (_message.Message,), {
    'DESCRIPTOR': _REPLICATEMODELRESP,
    '__module__': 'rlearn.distributed.actor_pb2'
    # @@protoc_insertion_point(class_scope:actor.ReplicateModelResp)
})
_sym_db.RegisterMessage(ReplicateModelResp)

TerminateReq = _reflection.GeneratedProtocolMessageType('TerminateReq', (_message.Message,), {
    'DESCRIPTOR': _TERMINATEREQ,
    '__module__': 'rlearn.distributed.actor_pb2'
    # @@protoc_insertion_point(class_scope:actor.TerminateReq)
})
_sym_db.RegisterMessage(TerminateReq)

TerminateResp = _reflection.GeneratedProtocolMessageType('TerminateResp', (_message.Message,), {
    'DESCRIPTOR': _TERMINATERESP,
    '__module__': 'rlearn.distributed.actor_pb2'
    # @@protoc_insertion_point(class_scope:actor.TerminateResp)
})
_sym_db.RegisterMessage(TerminateResp)

_ACTOR = DESCRIPTOR.services_by_name['Actor']
if _descriptor._USE_C_DESCRIPTORS == False:
    DESCRIPTOR._options = None
    DESCRIPTOR._serialized_options = b'\n\024io.grpc.examples.envB\nActorProtoP\001\242\002\003HLW'
    _SERVICEREADYREQ._serialized_start = 41
    _SERVICEREADYREQ._serialized_end = 77
    _SERVICEREADYRESP._serialized_start = 79
    _SERVICEREADYRESP._serialized_end = 131
    _STARTMETA._serialized_start = 134
    _STARTMETA._serialized_end = 329
    _STARTREQ._serialized_start = 331
    _STARTREQ._serialized_end = 407
    _STARTRESP._serialized_start = 409
    _STARTRESP._serialized_end = 466
    _MODELMETA._serialized_start = 468
    _MODELMETA._serialized_end = 515
    _GETGRADIENTSREQ._serialized_start = 517
    _GETGRADIENTSREQ._serialized_end = 553
    _GETGRADIENTSRESP._serialized_start = 555
    _GETGRADIENTSRESP._serialized_end = 639
    _REPLICATEMODELREQ._serialized_start = 641
    _REPLICATEMODELREQ._serialized_end = 726
    _REPLICATEMODELRESP._serialized_start = 728
    _REPLICATEMODELRESP._serialized_end = 794
    _TERMINATEREQ._serialized_start = 796
    _TERMINATEREQ._serialized_end = 829
    _TERMINATERESP._serialized_start = 831
    _TERMINATERESP._serialized_end = 892
    _ACTOR._serialized_start = 895
    _ACTOR._serialized_end = 1219
# @@protoc_insertion_point(module_scope)