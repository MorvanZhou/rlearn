# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: rlearn/distribute/buffer.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='rlearn/distribute/buffer.proto',
  package='replayBuffer',
  syntax='proto3',
  serialized_options=b'\n\024io.grpc.examples.envB\021ReplayBufferProtoP\001\242\002\003HLW',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x1erlearn/distribute/buffer.proto\x12\x0creplayBuffer\"$\n\x0fServiceReadyReq\x12\x11\n\trequestId\x18\x01 \x01(\t\"4\n\x10ServiceReadyResp\x12\r\n\x05ready\x18\x01 \x01(\x08\x12\x11\n\trequestId\x18\x02 \x01(\t\"*\n\x04\x44\x61ta\x12\x0e\n\x06values\x18\x01 \x03(\x02\x12\x12\n\nattributes\x18\x02 \x01(\t\"D\n\rUploadDataReq\x12 \n\x04\x64\x61ta\x18\x01 \x01(\x0b\x32\x12.replayBuffer.Data\x12\x11\n\trequestId\x18\x02 \x01(\t\">\n\x0eUploadDataResp\x12\x0c\n\x04\x64one\x18\x01 \x01(\x08\x12\x0b\n\x03\x65rr\x18\x02 \x01(\t\x12\x11\n\trequestId\x18\x03 \x01(\t\"5\n\x0f\x44ownloadDataReq\x12\x0f\n\x07maxSize\x18\x01 \x01(\x05\x12\x11\n\trequestId\x18\x02 \x01(\t\"T\n\x10\x44ownloadDataResp\x12 \n\x04\x64\x61ta\x18\x01 \x01(\x0b\x32\x12.replayBuffer.Data\x12\x0b\n\x03\x65rr\x18\x02 \x01(\t\x12\x11\n\trequestId\x18\x03 \x01(\t2\xfb\x01\n\x0cReplayBuffer\x12O\n\x0cServiceReady\x12\x1d.replayBuffer.ServiceReadyReq\x1a\x1e.replayBuffer.ServiceReadyResp\"\x00\x12I\n\nUploadData\x12\x1b.replayBuffer.UploadDataReq\x1a\x1c.replayBuffer.UploadDataResp\"\x00\x12O\n\x0c\x44ownloadData\x12\x1d.replayBuffer.DownloadDataReq\x1a\x1e.replayBuffer.DownloadDataResp\"\x00\x42\x31\n\x14io.grpc.examples.envB\x11ReplayBufferProtoP\x01\xa2\x02\x03HLWb\x06proto3'
)




_SERVICEREADYREQ = _descriptor.Descriptor(
  name='ServiceReadyReq',
  full_name='replayBuffer.ServiceReadyReq',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='requestId', full_name='replayBuffer.ServiceReadyReq.requestId', index=0,
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
  serialized_start=48,
  serialized_end=84,
)


_SERVICEREADYRESP = _descriptor.Descriptor(
  name='ServiceReadyResp',
  full_name='replayBuffer.ServiceReadyResp',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='ready', full_name='replayBuffer.ServiceReadyResp.ready', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='requestId', full_name='replayBuffer.ServiceReadyResp.requestId', index=1,
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
  serialized_start=86,
  serialized_end=138,
)


_DATA = _descriptor.Descriptor(
  name='Data',
  full_name='replayBuffer.Data',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='values', full_name='replayBuffer.Data.values', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='attributes', full_name='replayBuffer.Data.attributes', index=1,
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
  serialized_start=140,
  serialized_end=182,
)


_UPLOADDATAREQ = _descriptor.Descriptor(
  name='UploadDataReq',
  full_name='replayBuffer.UploadDataReq',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='data', full_name='replayBuffer.UploadDataReq.data', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='requestId', full_name='replayBuffer.UploadDataReq.requestId', index=1,
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
  serialized_start=184,
  serialized_end=252,
)


_UPLOADDATARESP = _descriptor.Descriptor(
  name='UploadDataResp',
  full_name='replayBuffer.UploadDataResp',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='done', full_name='replayBuffer.UploadDataResp.done', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='err', full_name='replayBuffer.UploadDataResp.err', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='requestId', full_name='replayBuffer.UploadDataResp.requestId', index=2,
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
  serialized_start=254,
  serialized_end=316,
)


_DOWNLOADDATAREQ = _descriptor.Descriptor(
  name='DownloadDataReq',
  full_name='replayBuffer.DownloadDataReq',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='maxSize', full_name='replayBuffer.DownloadDataReq.maxSize', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='requestId', full_name='replayBuffer.DownloadDataReq.requestId', index=1,
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
  serialized_start=318,
  serialized_end=371,
)


_DOWNLOADDATARESP = _descriptor.Descriptor(
  name='DownloadDataResp',
  full_name='replayBuffer.DownloadDataResp',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='data', full_name='replayBuffer.DownloadDataResp.data', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='err', full_name='replayBuffer.DownloadDataResp.err', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='requestId', full_name='replayBuffer.DownloadDataResp.requestId', index=2,
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
  serialized_start=373,
  serialized_end=457,
)

_UPLOADDATAREQ.fields_by_name['data'].message_type = _DATA
_DOWNLOADDATARESP.fields_by_name['data'].message_type = _DATA
DESCRIPTOR.message_types_by_name['ServiceReadyReq'] = _SERVICEREADYREQ
DESCRIPTOR.message_types_by_name['ServiceReadyResp'] = _SERVICEREADYRESP
DESCRIPTOR.message_types_by_name['Data'] = _DATA
DESCRIPTOR.message_types_by_name['UploadDataReq'] = _UPLOADDATAREQ
DESCRIPTOR.message_types_by_name['UploadDataResp'] = _UPLOADDATARESP
DESCRIPTOR.message_types_by_name['DownloadDataReq'] = _DOWNLOADDATAREQ
DESCRIPTOR.message_types_by_name['DownloadDataResp'] = _DOWNLOADDATARESP
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ServiceReadyReq = _reflection.GeneratedProtocolMessageType('ServiceReadyReq', (_message.Message,), {
  'DESCRIPTOR' : _SERVICEREADYREQ,
  '__module__' : 'rlearn.distribute.buffer_pb2'
  # @@protoc_insertion_point(class_scope:replayBuffer.ServiceReadyReq)
  })
_sym_db.RegisterMessage(ServiceReadyReq)

ServiceReadyResp = _reflection.GeneratedProtocolMessageType('ServiceReadyResp', (_message.Message,), {
  'DESCRIPTOR' : _SERVICEREADYRESP,
  '__module__' : 'rlearn.distribute.buffer_pb2'
  # @@protoc_insertion_point(class_scope:replayBuffer.ServiceReadyResp)
  })
_sym_db.RegisterMessage(ServiceReadyResp)

Data = _reflection.GeneratedProtocolMessageType('Data', (_message.Message,), {
  'DESCRIPTOR' : _DATA,
  '__module__' : 'rlearn.distribute.buffer_pb2'
  # @@protoc_insertion_point(class_scope:replayBuffer.Data)
  })
_sym_db.RegisterMessage(Data)

UploadDataReq = _reflection.GeneratedProtocolMessageType('UploadDataReq', (_message.Message,), {
  'DESCRIPTOR' : _UPLOADDATAREQ,
  '__module__' : 'rlearn.distribute.buffer_pb2'
  # @@protoc_insertion_point(class_scope:replayBuffer.UploadDataReq)
  })
_sym_db.RegisterMessage(UploadDataReq)

UploadDataResp = _reflection.GeneratedProtocolMessageType('UploadDataResp', (_message.Message,), {
  'DESCRIPTOR' : _UPLOADDATARESP,
  '__module__' : 'rlearn.distribute.buffer_pb2'
  # @@protoc_insertion_point(class_scope:replayBuffer.UploadDataResp)
  })
_sym_db.RegisterMessage(UploadDataResp)

DownloadDataReq = _reflection.GeneratedProtocolMessageType('DownloadDataReq', (_message.Message,), {
  'DESCRIPTOR' : _DOWNLOADDATAREQ,
  '__module__' : 'rlearn.distribute.buffer_pb2'
  # @@protoc_insertion_point(class_scope:replayBuffer.DownloadDataReq)
  })
_sym_db.RegisterMessage(DownloadDataReq)

DownloadDataResp = _reflection.GeneratedProtocolMessageType('DownloadDataResp', (_message.Message,), {
  'DESCRIPTOR' : _DOWNLOADDATARESP,
  '__module__' : 'rlearn.distribute.buffer_pb2'
  # @@protoc_insertion_point(class_scope:replayBuffer.DownloadDataResp)
  })
_sym_db.RegisterMessage(DownloadDataResp)


DESCRIPTOR._options = None

_REPLAYBUFFER = _descriptor.ServiceDescriptor(
  name='ReplayBuffer',
  full_name='replayBuffer.ReplayBuffer',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=460,
  serialized_end=711,
  methods=[
  _descriptor.MethodDescriptor(
    name='ServiceReady',
    full_name='replayBuffer.ReplayBuffer.ServiceReady',
    index=0,
    containing_service=None,
    input_type=_SERVICEREADYREQ,
    output_type=_SERVICEREADYRESP,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='UploadData',
    full_name='replayBuffer.ReplayBuffer.UploadData',
    index=1,
    containing_service=None,
    input_type=_UPLOADDATAREQ,
    output_type=_UPLOADDATARESP,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='DownloadData',
    full_name='replayBuffer.ReplayBuffer.DownloadData',
    index=2,
    containing_service=None,
    input_type=_DOWNLOADDATAREQ,
    output_type=_DOWNLOADDATARESP,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_REPLAYBUFFER)

DESCRIPTOR.services_by_name['ReplayBuffer'] = _REPLAYBUFFER

# @@protoc_insertion_point(module_scope)
