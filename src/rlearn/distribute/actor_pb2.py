# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: rlearn/distribute/actor.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()

DESCRIPTOR = _descriptor.FileDescriptor(
    name='rlearn/distribute/actor.proto',
    package='actor',
    syntax='proto3',
    serialized_options=b'\n\024io.grpc.examples.envB\021ReplayBufferProtoP\001\242\002\003HLW',
    create_key=_descriptor._internal_create_key,
    serialized_pb=b'\n\x1drlearn/distribute/actor.proto\x12\x05\x61\x63tor\"$\n\x0fServiceReadyReq\x12\x11\n\trequestId\x18\x01 \x01(\t\"4\n\x10ServiceReadyResp\x12\r\n\x05ready\x18\x01 \x01(\x08\x12\x11\n\trequestId\x18\x02 \x01(\t\"\x80\x01\n\tStartMeta\x12\x10\n\x08\x66ilename\x18\x01 \x01(\t\x12\x11\n\tmodelType\x18\x02 \x01(\t\x12\x12\n\nmaxEpisode\x18\x03 \x01(\x03\x12\x16\n\x0emaxEpisodeStep\x18\x04 \x01(\x03\x12\x0f\n\x07version\x18\x05 \x01(\t\x12\x11\n\trequestId\x18\x06 \x01(\t\"L\n\x08StartReq\x12 \n\x04meta\x18\x01 \x01(\x0b\x32\x10.actor.StartMetaH\x00\x12\x13\n\tchunkData\x18\x02 \x01(\x0cH\x00\x42\t\n\x07request\"9\n\tStartResp\x12\x0c\n\x04\x64one\x18\x01 \x01(\x08\x12\x0b\n\x03\x65rr\x18\x02 \x01(\t\x12\x11\n\trequestId\x18\x03 \x01(\t\"J\n\x12ReplicateModelMeta\x12\x10\n\x08\x66ilename\x18\x01 \x01(\t\x12\x0f\n\x07version\x18\x02 \x01(\t\x12\x11\n\trequestId\x18\x03 \x01(\t\"^\n\x11ReplicateModelReq\x12)\n\x04meta\x18\x01 \x01(\x0b\x32\x19.actor.ReplicateModelMetaH\x00\x12\x13\n\tchunkData\x18\x02 \x01(\x0cH\x00\x42\t\n\x07request\"B\n\x12ReplicateModelResp\x12\x0c\n\x04\x64one\x18\x01 \x01(\x08\x12\x0b\n\x03\x65rr\x18\x02 \x01(\t\x12\x11\n\trequestId\x18\x03 \x01(\t\"!\n\x0cTerminateReq\x12\x11\n\trequestId\x18\x01 \x01(\t\"=\n\rTerminateResp\x12\x0c\n\x04\x64one\x18\x01 \x01(\x08\x12\x0b\n\x03\x65rr\x18\x02 \x01(\t\x12\x11\n\trequestId\x18\x03 \x01(\t2\xff\x01\n\x05\x41\x63tor\x12\x41\n\x0cServiceReady\x12\x16.actor.ServiceReadyReq\x1a\x17.actor.ServiceReadyResp\"\x00\x12.\n\x05Start\x12\x0f.actor.StartReq\x1a\x10.actor.StartResp\"\x00(\x01\x12I\n\x0eReplicateModel\x12\x18.actor.ReplicateModelReq\x1a\x19.actor.ReplicateModelResp\"\x00(\x01\x12\x38\n\tTerminate\x12\x13.actor.TerminateReq\x1a\x14.actor.TerminateResp\"\x00\x42\x31\n\x14io.grpc.examples.envB\x11ReplayBufferProtoP\x01\xa2\x02\x03HLWb\x06proto3'
)

_SERVICEREADYREQ = _descriptor.Descriptor(
    name='ServiceReadyReq',
    full_name='actor.ServiceReadyReq',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='requestId', full_name='actor.ServiceReadyReq.requestId', index=0,
            number=1, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=b"".decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=40,
    serialized_end=76,
)


_SERVICEREADYRESP = _descriptor.Descriptor(
    name='ServiceReadyResp',
    full_name='actor.ServiceReadyResp',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='ready', full_name='actor.ServiceReadyResp.ready', index=0,
            number=1, type=8, cpp_type=7, label=1,
            has_default_value=False, default_value=False,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='requestId', full_name='actor.ServiceReadyResp.requestId', index=1,
            number=2, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=b"".decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=78,
    serialized_end=130,
)

_STARTMETA = _descriptor.Descriptor(
    name='StartMeta',
    full_name='actor.StartMeta',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='filename', full_name='actor.StartMeta.filename', index=0,
            number=1, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=b"".decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='modelType', full_name='actor.StartMeta.modelType', index=1,
            number=2, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=b"".decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='maxEpisode', full_name='actor.StartMeta.maxEpisode', index=2,
            number=3, type=3, cpp_type=2, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='maxEpisodeStep', full_name='actor.StartMeta.maxEpisodeStep', index=3,
            number=4, type=3, cpp_type=2, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='version', full_name='actor.StartMeta.version', index=4,
            number=5, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=b"".decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='requestId', full_name='actor.StartMeta.requestId', index=5,
            number=6, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=b"".decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=133,
    serialized_end=261,
)

_STARTREQ = _descriptor.Descriptor(
    name='StartReq',
    full_name='actor.StartReq',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='meta', full_name='actor.StartReq.meta', index=0,
            number=1, type=11, cpp_type=10, label=1,
            has_default_value=False, default_value=None,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='chunkData', full_name='actor.StartReq.chunkData', index=1,
            number=2, type=12, cpp_type=9, label=1,
            has_default_value=False, default_value=b"",
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
        _descriptor.OneofDescriptor(
            name='request', full_name='actor.StartReq.request',
            index=0, containing_type=None,
            create_key=_descriptor._internal_create_key,
            fields=[]),
    ],
    serialized_start=263,
    serialized_end=339,
)


_STARTRESP = _descriptor.Descriptor(
    name='StartResp',
    full_name='actor.StartResp',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='done', full_name='actor.StartResp.done', index=0,
            number=1, type=8, cpp_type=7, label=1,
            has_default_value=False, default_value=False,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='err', full_name='actor.StartResp.err', index=1,
            number=2, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=b"".decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='requestId', full_name='actor.StartResp.requestId', index=2,
            number=3, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=b"".decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=341,
    serialized_end=398,
)

_REPLICATEMODELMETA = _descriptor.Descriptor(
    name='ReplicateModelMeta',
    full_name='actor.ReplicateModelMeta',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='filename', full_name='actor.ReplicateModelMeta.filename', index=0,
            number=1, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=b"".decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='version', full_name='actor.ReplicateModelMeta.version', index=1,
            number=2, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=b"".decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='requestId', full_name='actor.ReplicateModelMeta.requestId', index=2,
            number=3, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=b"".decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=400,
    serialized_end=474,
)

_REPLICATEMODELREQ = _descriptor.Descriptor(
    name='ReplicateModelReq',
    full_name='actor.ReplicateModelReq',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='meta', full_name='actor.ReplicateModelReq.meta', index=0,
            number=1, type=11, cpp_type=10, label=1,
            has_default_value=False, default_value=None,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='chunkData', full_name='actor.ReplicateModelReq.chunkData', index=1,
            number=2, type=12, cpp_type=9, label=1,
            has_default_value=False, default_value=b"",
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
        _descriptor.OneofDescriptor(
            name='request', full_name='actor.ReplicateModelReq.request',
            index=0, containing_type=None,
            create_key=_descriptor._internal_create_key,
            fields=[]),
    ],
    serialized_start=476,
    serialized_end=570,
)


_REPLICATEMODELRESP = _descriptor.Descriptor(
    name='ReplicateModelResp',
    full_name='actor.ReplicateModelResp',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='done', full_name='actor.ReplicateModelResp.done', index=0,
            number=1, type=8, cpp_type=7, label=1,
            has_default_value=False, default_value=False,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='err', full_name='actor.ReplicateModelResp.err', index=1,
            number=2, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=b"".decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='requestId', full_name='actor.ReplicateModelResp.requestId', index=2,
            number=3, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=b"".decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=572,
    serialized_end=638,
)


_TERMINATEREQ = _descriptor.Descriptor(
    name='TerminateReq',
    full_name='actor.TerminateReq',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='requestId', full_name='actor.TerminateReq.requestId', index=0,
            number=1, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=b"".decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=640,
    serialized_end=673,
)


_TERMINATERESP = _descriptor.Descriptor(
    name='TerminateResp',
    full_name='actor.TerminateResp',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='done', full_name='actor.TerminateResp.done', index=0,
            number=1, type=8, cpp_type=7, label=1,
            has_default_value=False, default_value=False,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='err', full_name='actor.TerminateResp.err', index=1,
            number=2, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=b"".decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
        _descriptor.FieldDescriptor(
            name='requestId', full_name='actor.TerminateResp.requestId', index=2,
            number=3, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=b"".decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            serialized_options=None, file=DESCRIPTOR, create_key=_descriptor._internal_create_key),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=675,
    serialized_end=736,
)

_STARTREQ.fields_by_name['meta'].message_type = _STARTMETA
_STARTREQ.oneofs_by_name['request'].fields.append(
    _STARTREQ.fields_by_name['meta'])
_STARTREQ.fields_by_name['meta'].containing_oneof = _STARTREQ.oneofs_by_name['request']
_STARTREQ.oneofs_by_name['request'].fields.append(
    _STARTREQ.fields_by_name['chunkData'])
_STARTREQ.fields_by_name['chunkData'].containing_oneof = _STARTREQ.oneofs_by_name['request']
_REPLICATEMODELREQ.fields_by_name['meta'].message_type = _REPLICATEMODELMETA
_REPLICATEMODELREQ.oneofs_by_name['request'].fields.append(
    _REPLICATEMODELREQ.fields_by_name['meta'])
_REPLICATEMODELREQ.fields_by_name['meta'].containing_oneof = _REPLICATEMODELREQ.oneofs_by_name['request']
_REPLICATEMODELREQ.oneofs_by_name['request'].fields.append(
    _REPLICATEMODELREQ.fields_by_name['chunkData'])
_REPLICATEMODELREQ.fields_by_name['chunkData'].containing_oneof = _REPLICATEMODELREQ.oneofs_by_name['request']
DESCRIPTOR.message_types_by_name['ServiceReadyReq'] = _SERVICEREADYREQ
DESCRIPTOR.message_types_by_name['ServiceReadyResp'] = _SERVICEREADYRESP
DESCRIPTOR.message_types_by_name['StartMeta'] = _STARTMETA
DESCRIPTOR.message_types_by_name['StartReq'] = _STARTREQ
DESCRIPTOR.message_types_by_name['StartResp'] = _STARTRESP
DESCRIPTOR.message_types_by_name['ReplicateModelMeta'] = _REPLICATEMODELMETA
DESCRIPTOR.message_types_by_name['ReplicateModelReq'] = _REPLICATEMODELREQ
DESCRIPTOR.message_types_by_name['ReplicateModelResp'] = _REPLICATEMODELRESP
DESCRIPTOR.message_types_by_name['TerminateReq'] = _TERMINATEREQ
DESCRIPTOR.message_types_by_name['TerminateResp'] = _TERMINATERESP
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ServiceReadyReq = _reflection.GeneratedProtocolMessageType('ServiceReadyReq', (_message.Message,), {
    'DESCRIPTOR': _SERVICEREADYREQ,
    '__module__': 'rlearn.distribute.actor_pb2'
    # @@protoc_insertion_point(class_scope:actor.ServiceReadyReq)
})
_sym_db.RegisterMessage(ServiceReadyReq)

ServiceReadyResp = _reflection.GeneratedProtocolMessageType('ServiceReadyResp', (_message.Message,), {
    'DESCRIPTOR': _SERVICEREADYRESP,
    '__module__': 'rlearn.distribute.actor_pb2'
    # @@protoc_insertion_point(class_scope:actor.ServiceReadyResp)
})
_sym_db.RegisterMessage(ServiceReadyResp)

StartMeta = _reflection.GeneratedProtocolMessageType('StartMeta', (_message.Message,), {
    'DESCRIPTOR': _STARTMETA,
    '__module__': 'rlearn.distribute.actor_pb2'
    # @@protoc_insertion_point(class_scope:actor.StartMeta)
})
_sym_db.RegisterMessage(StartMeta)

StartReq = _reflection.GeneratedProtocolMessageType('StartReq', (_message.Message,), {
    'DESCRIPTOR': _STARTREQ,
    '__module__': 'rlearn.distribute.actor_pb2'
    # @@protoc_insertion_point(class_scope:actor.StartReq)
})
_sym_db.RegisterMessage(StartReq)

StartResp = _reflection.GeneratedProtocolMessageType('StartResp', (_message.Message,), {
    'DESCRIPTOR': _STARTRESP,
    '__module__': 'rlearn.distribute.actor_pb2'
    # @@protoc_insertion_point(class_scope:actor.StartResp)
})
_sym_db.RegisterMessage(StartResp)

ReplicateModelMeta = _reflection.GeneratedProtocolMessageType('ReplicateModelMeta', (_message.Message,), {
    'DESCRIPTOR': _REPLICATEMODELMETA,
    '__module__': 'rlearn.distribute.actor_pb2'
    # @@protoc_insertion_point(class_scope:actor.ReplicateModelMeta)
})
_sym_db.RegisterMessage(ReplicateModelMeta)

ReplicateModelReq = _reflection.GeneratedProtocolMessageType('ReplicateModelReq', (_message.Message,), {
    'DESCRIPTOR': _REPLICATEMODELREQ,
    '__module__': 'rlearn.distribute.actor_pb2'
    # @@protoc_insertion_point(class_scope:actor.ReplicateModelReq)
})
_sym_db.RegisterMessage(ReplicateModelReq)

ReplicateModelResp = _reflection.GeneratedProtocolMessageType('ReplicateModelResp', (_message.Message,), {
    'DESCRIPTOR': _REPLICATEMODELRESP,
    '__module__': 'rlearn.distribute.actor_pb2'
    # @@protoc_insertion_point(class_scope:actor.ReplicateModelResp)
})
_sym_db.RegisterMessage(ReplicateModelResp)

TerminateReq = _reflection.GeneratedProtocolMessageType('TerminateReq', (_message.Message,), {
    'DESCRIPTOR': _TERMINATEREQ,
    '__module__': 'rlearn.distribute.actor_pb2'
    # @@protoc_insertion_point(class_scope:actor.TerminateReq)
})
_sym_db.RegisterMessage(TerminateReq)

TerminateResp = _reflection.GeneratedProtocolMessageType('TerminateResp', (_message.Message,), {
    'DESCRIPTOR': _TERMINATERESP,
    '__module__': 'rlearn.distribute.actor_pb2'
    # @@protoc_insertion_point(class_scope:actor.TerminateResp)
})
_sym_db.RegisterMessage(TerminateResp)

DESCRIPTOR._options = None

_ACTOR = _descriptor.ServiceDescriptor(
    name='Actor',
    full_name='actor.Actor',
    file=DESCRIPTOR,
    index=0,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_start=739,
    serialized_end=994,
    methods=[
        _descriptor.MethodDescriptor(
            name='ServiceReady',
            full_name='actor.Actor.ServiceReady',
            index=0,
            containing_service=None,
            input_type=_SERVICEREADYREQ,
            output_type=_SERVICEREADYRESP,
            serialized_options=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.MethodDescriptor(
            name='Start',
            full_name='actor.Actor.Start',
            index=1,
            containing_service=None,
            input_type=_STARTREQ,
            output_type=_STARTRESP,
            serialized_options=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.MethodDescriptor(
            name='ReplicateModel',
            full_name='actor.Actor.ReplicateModel',
            index=2,
            containing_service=None,
            input_type=_REPLICATEMODELREQ,
            output_type=_REPLICATEMODELRESP,
            serialized_options=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.MethodDescriptor(
            name='Terminate',
            full_name='actor.Actor.Terminate',
            index=3,
            containing_service=None,
            input_type=_TERMINATEREQ,
            output_type=_TERMINATERESP,
            serialized_options=None,
            create_key=_descriptor._internal_create_key,
        ),
    ])
_sym_db.RegisterServiceDescriptor(_ACTOR)

DESCRIPTOR.services_by_name['Actor'] = _ACTOR

# @@protoc_insertion_point(module_scope)
