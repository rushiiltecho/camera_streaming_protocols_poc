# Camera Streaming System Setup Instructions

## Prerequisites

- Python 3.7+
- OpenCV
- gRPC
- NumPy

## Installation

1. Install required packages:

```bash
pip install opencv-python grpcio grpcio-tools numpy protobuf
```

2. Generate Python code from the protobuf definitions:

```bash
# From the project root directory
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. camera.proto
```

This will generate two files:
- `camera_pb2.py` (message classes)
- `camera_pb2_grpc.py` (service classes)

## Running the Server

Start the camera server:

```bash
# Basic usage
python camera_server.py

# With custom port and worker threads
python camera_server.py --port 50051 --workers 10
```

## Running the Client

```bash
# List available cameras
python camera_client.py --list

# View a single camera (camera 0)
python camera_client.py

# View multiple cameras
python camera_client.py --cameras 0 1 2

# Connect to a different server
python camera_client.py --server 192.168.1.100:50051
```

## System Architecture

The system follows a client-server architecture where:

1. **Server** manages camera connections and provides:
   - Camera discovery
   - Single frame capture
   - Continuous streaming with error handling

2. **Client** provides:
   - Connection to the gRPC server
   - Frame buffering for multiple cameras
   - Display capabilities for multiple simultaneous feeds

## Performance Considerations

- The server uses thread pools to handle multiple clients
- Frame compression reduces bandwidth requirements
- Buffering on both client and server sides handles network latency
- Error recovery mechanisms for camera disconnections
- Automatic reconnection for failed camera connections

## Camera Support

The system supports any camera compatible with OpenCV's VideoCapture:
- USB webcams
- IP cameras via RTSP/HTTP
- Virtual cameras
- Video files (by providing a file path instead of a camera ID)

## Extending the System

To add support for more camera features:
1. Update the `camera.proto` file with new message types and RPC methods
2. Regenerate the Python code using protoc
3. Implement the new methods in the server
4. Update the client to use the new features