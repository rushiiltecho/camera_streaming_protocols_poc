# camera_server.py
import time
import cv2
import grpc
import numpy as np
import concurrent.futures
from threading import Lock
import proto.camera_pb2 as camera_pb2
import proto.camera_pb2_grpc as camera_pb2_grpc
from concurrent import futures
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CameraDevice:
    """Class to handle camera operations with error handling and reconnection logic."""
    
    def __init__(self, camera_id, width=640, height=480, fps=30):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.lock = Lock()
        self.connected = False
        self.last_frame = None
        self.connect()
    
    def connect(self):
        """Connect to the camera with retries."""
        try:
            with self.lock:
                if self.cap is not None:
                    self.cap.release()
                
                logger.info(f"Connecting to camera {self.camera_id}")
                self.cap = cv2.VideoCapture(self.camera_id)
                
                if not self.cap.isOpened():
                    logger.error(f"Failed to open camera {self.camera_id}")
                    self.connected = False
                    return False
                
                # Set camera properties
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                
                self.connected = True
                logger.info(f"Successfully connected to camera {self.camera_id}")
                return True
        except Exception as e:
            logger.error(f"Error connecting to camera {self.camera_id}: {str(e)}")
            self.connected = False
            return False
    
    def get_frame(self):
        """Get the current frame from the camera with error handling."""
        if not self.connected:
            if not self.connect():
                # Return blank frame if still not connected
                if self.last_frame is not None:
                    # Return the last valid frame with a warning indicator
                    frame = self.last_frame.copy()
                    cv2.putText(frame, "CAMERA DISCONNECTED", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    return frame
                else:
                    # Create blank frame with error message
                    frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    cv2.putText(frame, f"Camera {self.camera_id} disconnected", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    return frame
        
        try:
            with self.lock:
                if self.cap is None:
                    return None
                    
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.warning(f"Failed to read frame from camera {self.camera_id}")
                    self.connected = False
                    # Return last valid frame or blank frame
                    if self.last_frame is not None:
                        return self.last_frame
                    else:
                        blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                        cv2.putText(blank, f"No signal from camera {self.camera_id}", (50, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        return blank
                
                # Store current frame as last valid frame
                self.last_frame = frame
                return frame
        except Exception as e:
            logger.error(f"Error getting frame from camera {self.camera_id}: {str(e)}")
            self.connected = False
            # Return blank frame with error message
            blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(blank, f"Error: {str(e)}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return blank
    
    def release(self):
        """Release camera resources."""
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
                self.connected = False
                logger.info(f"Released camera {self.camera_id}")


class CameraServicer(camera_pb2_grpc.CameraServiceServicer):
    """gRPC servicer implementing the Camera service."""
    
    def __init__(self):
        self.cameras = {}  # Dict to store camera devices
        self.camera_locks = {}  # Dict to store locks for each camera
    
    def get_camera(self, camera_id, width=640, height=480, fps=30):
        """Get or create a camera device."""
        if camera_id not in self.cameras:
            logger.info(f"Creating new camera instance for camera_id: {camera_id}")
            self.cameras[camera_id] = CameraDevice(camera_id, width, height, fps)
            self.camera_locks[camera_id] = Lock()
        return self.cameras[camera_id]
    
    def ListCameras(self, request, context):
        """List all available cameras."""
        try:
            # On most systems, cameras 0-9 are good to check
            available_cameras = []
            
            for i in range(10):
                # Try to open the camera briefly
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    # Get camera information
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    
                    available_cameras.append(
                        camera_pb2.CameraInfo(
                            id=i,
                            name=f"Camera {i}",
                            width=width,
                            height=height,
                            fps=fps
                        )
                    )
                    cap.release()
            
            response = camera_pb2.ListCamerasResponse(cameras=available_cameras)
            return response
        except Exception as e:
            logger.error(f"Error in ListCameras: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error listing cameras: {str(e)}")
            return camera_pb2.ListCamerasResponse()
    
    def GetCameraInfo(self, request, context):
        """Get information about a specific camera."""
        try:
            camera_id = request.id
            
            # Check if camera exists and is accessible
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Camera {camera_id} not found or not accessible")
                return camera_pb2.CameraInfo()
            
            # Get camera information
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            cap.release()
            
            # Return camera info
            return camera_pb2.CameraInfo(
                id=camera_id,
                name=f"Camera {camera_id}",
                width=width,
                height=height,
                fps=fps
            )
        except Exception as e:
            logger.error(f"Error in GetCameraInfo: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error getting camera info: {str(e)}")
            return camera_pb2.CameraInfo()
    
    def GetFrame(self, request, context):
        """Get a single frame from the specified camera."""
        try:
            camera_id = request.camera_id
            width = request.width if request.width > 0 else 640
            height = request.height if request.height > 0 else 480
            fps = request.fps if request.fps > 0 else 30
            
            camera = self.get_camera(camera_id, width, height, fps)
            
            # Acquire lock for this camera
            with self.camera_locks.get(camera_id, Lock()):
                frame = camera.get_frame()
                
                if frame is None:
                    context.set_code(grpc.StatusCode.UNAVAILABLE)
                    context.set_details(f"Camera {camera_id} unavailable")
                    return camera_pb2.FrameResponse()
                
                # Encode frame as JPEG
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                _, jpeg_frame = cv2.imencode('.jpg', frame, encode_param)
                
                # Create response
                timestamp_ms = int(time.time() * 1000)
                response = camera_pb2.FrameResponse(
                    camera_id=camera_id,
                    timestamp=timestamp_ms,
                    width=frame.shape[1],
                    height=frame.shape[0],
                    format="jpeg",
                    data=jpeg_frame.tobytes()
                )
                return response
        except Exception as e:
            logger.error(f"Error in GetFrame: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error getting frame: {str(e)}")
            return camera_pb2.FrameResponse()
    
    def StreamFrames(self, request, context):
        """Stream frames from the specified camera."""
        try:
            camera_id = request.camera_id
            width = request.width if request.width > 0 else 640
            height = request.height if request.height > 0 else 480
            fps = request.fps if request.fps > 0 else 30
            quality = request.quality if 0 < request.quality <= 100 else 90
            
            camera = self.get_camera(camera_id, width, height, fps)
            
            # Set up sleep time based on fps
            sleep_time = 1.0 / fps
            
            logger.info(f"Starting stream for camera {camera_id} at {fps} FPS")
            
            # Return frames until client disconnects
            while context.is_active():
                try:
                    frame = camera.get_frame()
                    
                    if frame is not None:
                        # Encode frame as JPEG
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                        _, jpeg_frame = cv2.imencode('.jpg', frame, encode_param)
                        
                        # Create response
                        timestamp_ms = int(time.time() * 1000)
                        response = camera_pb2.FrameResponse(
                            camera_id=camera_id,
                            timestamp=timestamp_ms,
                            width=frame.shape[1],
                            height=frame.shape[0],
                            format="jpeg",
                            data=jpeg_frame.tobytes()
                        )
                        yield response
                    
                    # Sleep to maintain frame rate
                    time.sleep(sleep_time)
                    
                except Exception as e:
                    logger.error(f"Error during streaming for camera {camera_id}: {str(e)}")
                    error_response = camera_pb2.FrameResponse(
                        camera_id=camera_id,
                        error=f"Stream error: {str(e)}"
                    )
                    yield error_response
                    time.sleep(1)  # Wait before retry
            
            logger.info(f"Stream ended for camera {camera_id}")
            
        except Exception as e:
            logger.error(f"Error in StreamFrames setup: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error setting up stream: {str(e)}")
    
    def cleanup(self):
        """Clean up all camera resources."""
        for camera_id, camera in self.cameras.items():
            logger.info(f"Cleaning up camera {camera_id}")
            camera.release()


def serve(port=50051, max_workers=10):
    """Start the gRPC server."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers),
                         options=[
                             ('grpc.max_send_message_length', 50 * 1024 * 1024),
                             ('grpc.max_receive_message_length', 50 * 1024 * 1024)
                         ])
    
    servicer = CameraServicer()
    camera_pb2_grpc.add_CameraServiceServicer_to_server(servicer, server)
    
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    
    logger.info(f"Camera gRPC server started on port {port}")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
        servicer.cleanup()
        server.stop(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Camera gRPC Server')
    parser.add_argument('--port', type=int, default=50051, help='Port to listen on')
    parser.add_argument('--workers', type=int, default=10, help='Number of worker threads')
    args = parser.parse_args()
    
    serve(port=args.port, max_workers=args.workers)