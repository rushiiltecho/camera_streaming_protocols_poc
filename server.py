import grpc
from concurrent import futures
import threading
from collections import defaultdict

import proto.video_pb2 as video_pb2
import proto.video_pb2_grpc as video_pb2_grpc

class VideoStreamingServicer(video_pb2_grpc.VideoStreamingServicer):
    def __init__(self):
        self.locks = defaultdict(threading.Lock)  # Per-camera locks
        self.latest_frames = {}  # camera_id -> Frame
        self.conditions = defaultdict(threading.Condition)  # Per-camera conditions
        self.active_cameras = set()  # Set of active camera IDs

    def SendFrameStream(self, request_iterator, context):
        """Handle frame stream from camera (client-side streaming)"""
        try:
            camera_id = None
            for camera_frame in request_iterator:
                camera_id = camera_frame.camera_id
                with self.locks[camera_id]:
                    self.latest_frames[camera_id] = camera_frame.frame
                    self.active_cameras.add(camera_id)
                # Notify all waiting viewers for this camera
                with self.conditions[camera_id]:
                    self.conditions[camera_id].notify_all()
            
            # Clean up when stream ends
            if camera_id:
                with self.locks[camera_id]:
                    self.active_cameras.discard(camera_id)
                    if camera_id in self.latest_frames:
                        del self.latest_frames[camera_id]
            
            return video_pb2.UploadStatus(success=True, message="Stream completed")
        except Exception as e:
            return video_pb2.UploadStatus(success=False, message=str(e))

    def ReceiveFrameStream(self, request, context):
        """Handle viewer requests (server-side streaming)"""
        camera_id = request.camera_id
        last_timestamp = 0
        
        # Check if camera exists
        if camera_id not in self.active_cameras:
            context.abort(grpc.StatusCode.NOT_FOUND, f"Camera {camera_id} not found")
        
        while camera_id in self.active_cameras:
            with self.conditions[camera_id]:
                self.conditions[camera_id].wait(timeout=1)
                if (camera_id in self.latest_frames and 
                    self.latest_frames[camera_id].timestamp > last_timestamp):
                    last_timestamp = self.latest_frames[camera_id].timestamp
                    yield self.latest_frames[camera_id]

    def ListCameras(self, request, context):
        """List all active cameras"""
        return video_pb2.ListCamerasResponse(
            camera_ids=list(self.active_cameras)
        )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    video_pb2_grpc.add_VideoStreamingServicer_to_server(
        VideoStreamingServicer(), server
    )
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started on port 50051")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()