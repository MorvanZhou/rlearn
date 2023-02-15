python3 -m grpc_tools.protoc --proto_path ./src --python_out ./src --grpc_python_out ./src ./src/rlearn/distributed/experience/buffer.proto
python3 -m grpc_tools.protoc --proto_path ./src --python_out ./src --grpc_python_out ./src ./src/rlearn/distributed/experience/actor.proto
python3 -m grpc_tools.protoc --proto_path ./src --python_out ./src --grpc_python_out ./src ./src/rlearn/distributed/gradient/worker.proto
