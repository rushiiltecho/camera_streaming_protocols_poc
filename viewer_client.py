import grpc
import cv2
import numpy as np
import proto.video_pb2 as video_pb2
import proto.video_pb2_grpc as video_pb2_grpc

def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = video_pb2_grpc.VideoStreamingStub(channel)
    
    for frame in stub.ReceiveFrameStream(video_pb2.StreamRequest()):
        img_array = np.frombuffer(frame.image_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is not None:
            cv2.imshow('Live Stream', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()