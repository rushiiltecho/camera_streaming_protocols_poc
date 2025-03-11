import grpc
from concurrent import futures
import threading

import proto.video_pb2 as video_pb2
import proto.video_pb2_grpc as video_pb2_grpc

class VideoStreamingServicer(video_pb2_grpc.VideoStreamingServicer):
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame = None
        self.condition = threading.Condition()
        self.active_viewers = []

    def SendFrameStream(self, request_iterator, context):
        """Handle frame stream from camera (client-side streaming)"""
        try:
            for frame in request_iterator:
                with self.lock:
                    self.latest_frame = frame
                # Notify all waiting viewers
                with self.condition:
                    self.condition.notify_all()
            return video_pb2.UploadStatus(success=True, message="Stream completed")
        except Exception as e:
            return video_pb2.UploadStatus(success=False, message=str(e))

    def ReceiveFrameStream(self, request, context):
        """Handle viewer requests (server-side streaming)"""
        last_timestamp = 0
        while True:
            with self.condition:
                self.condition.wait()
                if self.latest_frame and self.latest_frame.timestamp > last_timestamp:
                    last_timestamp = self.latest_frame.timestamp
                    yield self.latest_frame

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    video_pb2_grpc.add_VideoStreamingServicer_to_server(
        VideoStreamingServicer(), server
    )
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()