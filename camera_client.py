import grpc
import cv2
import time
import proto.video_pb2 as video_pb2
import proto.video_pb2_grpc as video_pb2_grpc

def generate_frames():
    cap = cv2.VideoCapture(0)  # 0 for default camera
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Encode as JPEG (adjust quality as needed)
        _, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        yield video_pb2.Frame(
            image_data=jpeg.tobytes(),
            timestamp=int(time.time() * 1000)
        )

def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = video_pb2_grpc.VideoStreamingStub(channel)
    
    response = stub.SendFrameStream(generate_frames())
    print(f"Streaming completed: {response.message}")

if __name__ == '__main__':
    run()