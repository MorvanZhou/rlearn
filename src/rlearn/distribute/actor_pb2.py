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
  serialized_pb=b'\n\x1drlearn/distribute/actor.proto\x12\x05\x61\x63tor\"$\n\x0fServiceReadyReq\x12\x11\n\trequestId\x18\x01 \x01(\t\"4\n\x10ServiceReadyResp\x12\r\n\x05ready\x18\x01 \x01(\x08\x12\x11\n\trequestId\x18\x02 \x01(\t\"\x1d\n\x08StartReq\x12\x11\n\trequestId\x18\x01 \x01(\t\"9\n\tStartResp\x12\x0c\n\x04\x64one\x18\x01 \x01(\x08\x12\x0b\n\x03\x65rr\x18\x02 \x01(\t\x12\x11\n\trequestId\x18\x03 \x01(\t\"*\n\x04Meta\x12\x0f\n\x07version\x18\x03 \x01(\t\x12\x11\n\trequestId\x18\x04 \x01(\t\"P\n\x11ReplicateModelReq\x12\x1b\n\x04meta\x18\x01 \x01(\x0b\x32\x0b.actor.MetaH\x00\x12\x13\n\tchunkData\x18\x02 \x01(\x0cH\x00\x42\t\n\x07request\"B\n\x12ReplicateModelResp\x12\x0c\n\x04\x64one\x18\x01 \x01(\x08\x12\x0b\n\x03\x65rr\x18\x02 \x01(\t\x12\x11\n\trequestId\x18\x03 \x01(\t\"!\n\x0cTerminateReq\x12\x11\n\trequestId\x18\x01 \x01(\t\"=\n\rTerminateResp\x12\x0c\n\x04\x64one\x18\x01 \x01(\x08\x12\x0b\n\x03\x65rr\x18\x02 \x01(\t\x12\x11\n\trequestId\x18\x03 \x01(\t2\xfb\x01\n\x05\x41\x63tor\x12\x41\n\x0cServiceReady\x12\x16.actor.ServiceReadyReq\x1a\x17.actor.ServiceReadyResp\"\x00\x12,\n\x05Start\x12\x0f.actor.StartReq\x1a\x10.actor.StartResp\"\x00\x12G\n\x0eReplicateModel\x12\x18.actor.ReplicateModelReq\x1a\x19.actor.ReplicateModelResp\"\x00\x12\x38\n\tTerminate\x12\x13.actor.TerminateReq\x1a\x14.actor.TerminateResp\"\x00\x42\x31\n\x14io.grpc.examples.envB\x11ReplayBufferProtoP\x01\xa2\x02\x03HLWb\x06proto3'
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
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='requestId', full_name='actor.ServiceReadyResp.requestId', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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


_STARTREQ = _descriptor.Descriptor(
  name='StartReq',
  full_name='actor.StartReq',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='requestId', full_name='actor.StartReq.requestId', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=132,
  serialized_end=161,
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
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='err', full_name='actor.StartResp.err', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='requestId', full_name='actor.StartResp.requestId', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=163,
  serialized_end=220,
)


_META = _descriptor.Descriptor(
  name='Meta',
  full_name='actor.Meta',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='version', full_name='actor.Meta.version', index=0,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='requestId', full_name='actor.Meta.requestId', index=1,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=222,
  serialized_end=264,
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
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='chunkData', full_name='actor.ReplicateModelReq.chunkData', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=266,
  serialized_end=346,
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
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='err', full_name='actor.ReplicateModelResp.err', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='requestId', full_name='actor.ReplicateModelResp.requestId', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=348,
  serialized_end=414,
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
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=416,
  serialized_end=449,
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
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='err', full_name='actor.TerminateResp.err', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='requestId', full_name='actor.TerminateResp.requestId', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=451,
  serialized_end=512,
)

_REPLICATEMODELREQ.fields_by_name['meta'].message_type = _META
_REPLICATEMODELREQ.oneofs_by_name['request'].fields.append(
  _REPLICATEMODELREQ.fields_by_name['meta'])
_REPLICATEMODELREQ.fields_by_name['meta'].containing_oneof = _REPLICATEMODELREQ.oneofs_by_name['request']
_REPLICATEMODELREQ.oneofs_by_name['request'].fields.append(
  _REPLICATEMODELREQ.fields_by_name['chunkData'])
_REPLICATEMODELREQ.fields_by_name['chunkData'].containing_oneof = _REPLICATEMODELREQ.oneofs_by_name['request']
DESCRIPTOR.message_types_by_name['ServiceReadyReq'] = _SERVICEREADYREQ
DESCRIPTOR.message_types_by_name['ServiceReadyResp'] = _SERVICEREADYRESP
DESCRIPTOR.message_types_by_name['StartReq'] = _STARTREQ
DESCRIPTOR.message_types_by_name['StartResp'] = _STARTRESP
DESCRIPTOR.message_types_by_name['Meta'] = _META
DESCRIPTOR.message_types_by_name['ReplicateModelReq'] = _REPLICATEMODELREQ
DESCRIPTOR.message_types_by_name['ReplicateModelResp'] = _REPLICATEMODELRESP
DESCRIPTOR.message_types_by_name['TerminateReq'] = _TERMINATEREQ
DESCRIPTOR.message_types_by_name['TerminateResp'] = _TERMINATERESP
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ServiceReadyReq = _reflection.GeneratedProtocolMessageType('ServiceReadyReq', (_message.Message,), {
  'DESCRIPTOR' : _SERVICEREADYREQ,
  '__module__' : 'rlearn.distribute.actor_pb2'
  # @@protoc_insertion_point(class_scope:actor.ServiceReadyReq)
  })
_sym_db.RegisterMessage(ServiceReadyReq)

ServiceReadyResp = _reflection.GeneratedProtocolMessageType('ServiceReadyResp', (_message.Message,), {
  'DESCRIPTOR' : _SERVICEREADYRESP,
  '__module__' : 'rlearn.distribute.actor_pb2'
  # @@protoc_insertion_point(class_scope:actor.ServiceReadyResp)
  })
_sym_db.RegisterMessage(ServiceReadyResp)

StartReq = _reflection.GeneratedProtocolMessageType('StartReq', (_message.Message,), {
  'DESCRIPTOR' : _STARTREQ,
  '__module__' : 'rlearn.distribute.actor_pb2'
  # @@protoc_insertion_point(class_scope:actor.StartReq)
  })
_sym_db.RegisterMessage(StartReq)

StartResp = _reflection.GeneratedProtocolMessageType('StartResp', (_message.Message,), {
  'DESCRIPTOR' : _STARTRESP,
  '__module__' : 'rlearn.distribute.actor_pb2'
  # @@protoc_insertion_point(class_scope:actor.StartResp)
  })
_sym_db.RegisterMessage(StartResp)

Meta = _reflection.GeneratedProtocolMessageType('Meta', (_message.Message,), {
  'DESCRIPTOR' : _META,
  '__module__' : 'rlearn.distribute.actor_pb2'
  # @@protoc_insertion_point(class_scope:actor.Meta)
  })
_sym_db.RegisterMessage(Meta)

ReplicateModelReq = _reflection.GeneratedProtocolMessageType('ReplicateModelReq', (_message.Message,), {
  'DESCRIPTOR' : _REPLICATEMODELREQ,
  '__module__' : 'rlearn.distribute.actor_pb2'
  # @@protoc_insertion_point(class_scope:actor.ReplicateModelReq)
  })
_sym_db.RegisterMessage(ReplicateModelReq)

ReplicateModelResp = _reflection.GeneratedProtocolMessageType('ReplicateModelResp', (_message.Message,), {
  'DESCRIPTOR' : _REPLICATEMODELRESP,
  '__module__' : 'rlearn.distribute.actor_pb2'
  # @@protoc_insertion_point(class_scope:actor.ReplicateModelResp)
  })
_sym_db.RegisterMessage(ReplicateModelResp)

TerminateReq = _reflection.GeneratedProtocolMessageType('TerminateReq', (_message.Message,), {
  'DESCRIPTOR' : _TERMINATEREQ,
  '__module__' : 'rlearn.distribute.actor_pb2'
  # @@protoc_insertion_point(class_scope:actor.TerminateReq)
  })
_sym_db.RegisterMessage(TerminateReq)

TerminateResp = _reflection.GeneratedProtocolMessageType('TerminateResp', (_message.Message,), {
  'DESCRIPTOR' : _TERMINATERESP,
  '__module__' : 'rlearn.distribute.actor_pb2'
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
  serialized_start=515,
  serialized_end=766,
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
