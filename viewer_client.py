import grpc
import cv2
import numpy as np
import proto.video_pb2 as video_pb2
import proto.video_pb2_grpc as video_pb2_grpc
def list_available_cameras(server_address):
    """List all active cameras from the server"""
    try:
        channel = grpc.insecure_channel(server_address)
        stub = video_pb2_grpc.VideoStreamingStub(channel)
        response = stub.ListCameras(video_pb2.ListCamerasRequest())
        return response.camera_ids
    except grpc.RpcError as e:
        print(f"Error listing cameras: {e.details()}")
        return []

def run(server_address):
    try:
        # List available cameras
        cameras = list_available_cameras(server_address)
        if not cameras:
            print("No cameras available")
            return
        
        print("Available cameras:")
        for i, cam in enumerate(cameras):
            print(f"{i+1}. {cam}")
        
        # Select camera
        selection = int(input("Select camera number: ")) - 1
        camera_id = cameras[selection]
        
        # Start streaming
        channel = grpc.insecure_channel(server_address)
        stub = video_pb2_grpc.VideoStreamingStub(channel)
        print(f"Connecting to camera {camera_id}...")
        
        for frame in stub.ReceiveFrameStream(video_pb2.StreamRequest(camera_id=camera_id)):
            img_array = np.frombuffer(frame.image_data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is not None:
                cv2.imshow(f'Camera {camera_id}', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except grpc.RpcError as e:
        print(f"gRPC error: {e.code()}: {e.details()}")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        cv2.destroyAllWindows()
        print("Viewer closed")

if __name__ == '__main__':
    run('localhost:50051')