import grpc
import cv2
import time
import argparse
import proto.video_pb2 as video_pb2
import proto.video_pb2_grpc as video_pb2_grpc

def generate_frames(camera_id, source=0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open camera {source} for camera_id {camera_id}")
        return
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to capture frame from camera {source}")
                break
                
            # Encode as JPEG
            _, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            
            yield video_pb2.CameraFrame(
                camera_id=camera_id,
                frame=video_pb2.Frame(
                    image_data=jpeg.tobytes(),
                    timestamp=int(time.time() * 1000)
                )
            )
            
            # Control frame rate
            time.sleep(0.03)  # ~30 FPS
            
    except KeyboardInterrupt:
        print("\nStopping camera stream...")
    finally:
        cap.release()
        print(f"Camera {source} resources released")

def run(camera_id, server_address, source=0):
    try:
        channel = grpc.insecure_channel(server_address)
        stub = video_pb2_grpc.VideoStreamingStub(channel)
        print(f"Starting stream for camera {camera_id} (source {source}) to server at {server_address}")
        
        response = stub.SendFrameStream(generate_frames(camera_id, source))
        print(f"Streaming completed: {response.message}")
        
    except grpc.RpcError as e:
        print(f"gRPC error: {e.code()}: {e.details()}")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        print("Client shutdown")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera-id', required=True, help='Unique camera identifier')
    parser.add_argument('--source', type=int, default=0, help='Camera source index')
    parser.add_argument('--server', default='localhost:50051', help='Server address')
    args = parser.parse_args()

    camera_id = args.camera_id
    run(camera_id, args.server, camera_id)