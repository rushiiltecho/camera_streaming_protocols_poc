# camera_client.py
import sys
import cv2
import grpc
import numpy as np
import time
import threading
import argparse
import proto.camera_pb2 as camera_pb2
import proto.camera_pb2_grpc as camera_pb2_grpc
import logging
from queue import Queue, Full, Empty
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check if OpenCV is properly installed
try:
    cv2_version = cv2.__version__
    logger.info(f"OpenCV version: {cv2_version}")
except Exception as e:
    logger.error(f"OpenCV import error: {str(e)}")
    raise ImportError("Failed to import OpenCV. Please check your installation.")

class FrameBuffer:
    """Thread-safe buffer for storing the latest frames from each camera."""
    
    def __init__(self, max_size=5):
        self.buffer = {}  # Dictionary mapping camera_id to a Queue
        self.latest_frames = {}  # Keep track of the latest frame for each camera
        self.lock = threading.Lock()
        self.max_size = max_size
        self.frame_counters = {}  # Count frames received per camera
        self.last_frame_time = {}  # Track timing of the latest frame per camera
    
    def get_buffer(self, camera_id):
        """Get or create a buffer for the specified camera."""
        with self.lock:
            if camera_id not in self.buffer:
                self.buffer[camera_id] = Queue(maxsize=self.max_size)
                self.frame_counters[camera_id] = 0
                self.last_frame_time[camera_id] = time.time()
            return self.buffer[camera_id]
    
    def put_frame(self, camera_id, frame, timestamp):
        """Add a frame to the buffer for the specified camera."""
        # Validate frame before buffering
        if frame is None or frame.size == 0:
            logger.warning(f"Attempt to buffer invalid frame for camera {camera_id}")
            return False
            
        try:
            buffer = self.get_buffer(camera_id)
            
            # Update statistics
            with self.lock:
                self.frame_counters[camera_id] += 1
                now = time.time()
                time_since_last = now - self.last_frame_time[camera_id]
                if time_since_last > 5.0:  # Log if significant gap between frames
                    logger.info(f"Camera {camera_id}: {time_since_last:.1f}s since last frame")
                self.last_frame_time[camera_id] = now
                
                # Store latest frame directly for quick access
                self.latest_frames[camera_id] = (frame.copy(), timestamp)
            
            # Add to queue buffer (with thread safety)
            try:
                # If buffer is full, remove oldest frame
                if buffer.full():
                    try:
                        buffer.get_nowait()
                    except Empty:
                        pass  # Queue was emptied by another thread
                        
                buffer.put_nowait((frame, timestamp))
                return True
            except Full:
                logger.warning(f"Frame buffer full for camera {camera_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error buffering frame for camera {camera_id}: {str(e)}")
            return False
    
    def get_latest_frame(self, camera_id):
        """Get the latest frame for the specified camera."""
        # First try the direct latest_frames dict for speed
        with self.lock:
            if camera_id in self.latest_frames:
                return self.latest_frames[camera_id]
        
        # Fall back to queue if not in latest_frames
        buffer = self.get_buffer(camera_id)
        try:
            # Get the newest frame (might leave buffer empty)
            all_frames = []
            while not buffer.empty():
                try:
                    frame_data = buffer.get_nowait()
                    all_frames.append(frame_data)
                except Empty:
                    break
            
            if all_frames:
                # Return the most recent frame
                newest_frame = all_frames[-1]
                
                # Optionally put the frames back except the oldest ones
                if len(all_frames) > 1:
                    for frame_data in all_frames[1:]:
                        try:
                            buffer.put_nowait(frame_data)
                        except Full:
                            break
                            
                return newest_frame
                
            return None, None
        except Exception as e:
            logger.error(f"Error retrieving frame for camera {camera_id}: {str(e)}")
            return None, None
    
    def get_all_frames(self):
        """Get the latest frames for all cameras."""
        result = {}
        
        # First try the faster direct dictionary
        with self.lock:
            # Make a copy to avoid race conditions
            latest_copy = self.latest_frames.copy()
            
        for camera_id, frame_data in latest_copy.items():
            if frame_data and frame_data[0] is not None:
                result[camera_id] = frame_data
        
        # If we didn't get any frames from the direct method, try the queue method
        if not result:
            with self.lock:
                camera_ids = list(self.buffer.keys())
            
            for camera_id in camera_ids:
                frame, timestamp = self.get_latest_frame(camera_id)
                if frame is not None:
                    result[camera_id] = (frame, timestamp)
        
        return result
        
    def get_stats(self):
        """Get buffer statistics for debugging."""
        with self.lock:
            stats = {
                'cameras': len(self.buffer),
                'frame_counts': self.frame_counters.copy(),
                'buffer_sizes': {camera_id: buffer.qsize() for camera_id, buffer in self.buffer.items()},
                'last_frame_times': {camera_id: time.time() - t for camera_id, t in self.last_frame_time.items()}
            }
        return stats


class CameraClient:
    """Client for the Camera gRPC service."""
    
    def __init__(self, server_address='localhost:50051'):
        self.server_address = server_address
        self.channel = None
        self.stub = None
        self.frame_buffer = FrameBuffer(max_size=10)
        self.stream_threads = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.running = False
        self.connect()
    
    def connect(self):
        """Connect to the gRPC server."""
        try:
            self.channel = grpc.insecure_channel(
                self.server_address,
                options=[
                    ('grpc.max_send_message_length', 50 * 1024 * 1024),
                    ('grpc.max_receive_message_length', 50 * 1024 * 1024)
                ]
            )
            self.stub = camera_pb2_grpc.CameraServiceStub(self.channel)
            logger.info(f"Connected to server at {self.server_address}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to server: {str(e)}")
            return False
    
    def list_cameras(self):
        """List all available cameras."""
        try:
            response = self.stub.ListCameras(camera_pb2.ListCamerasRequest())
            return response.cameras
        except grpc.RpcError as e:
            logger.error(f"RPC error: {e.code()}: {e.details()}")
            return []
        except Exception as e:
            logger.error(f"Error listing cameras: {str(e)}")
            return []
    
    def get_camera_info(self, camera_id):
        """Get information about a specific camera."""
        try:
            response = self.stub.GetCameraInfo(camera_pb2.CameraInfoRequest(id=camera_id))
            return response
        except grpc.RpcError as e:
            logger.error(f"RPC error: {e.code()}: {e.details()}")
            return None
        except Exception as e:
            logger.error(f"Error getting camera info: {str(e)}")
            return None
    
    def get_frame(self, camera_id, width=0, height=0, fps=0):
        """Get a single frame from the specified camera."""
        try:
            request = camera_pb2.FrameRequest(
                camera_id=camera_id,
                width=width,
                height=height,
                fps=fps
            )
            response = self.stub.GetFrame(request)
            
            # Decode JPEG frame
            if response.data:
                frame_data = np.frombuffer(response.data, dtype=np.uint8)
                frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
                return frame, response.timestamp
            else:
                logger.warning(f"Empty frame data for camera {camera_id}")
                return None, None
        except grpc.RpcError as e:
            logger.error(f"RPC error: {e.code()}: {e.details()}")
            return None, None
        except Exception as e:
            logger.error(f"Error getting frame: {str(e)}")
            return None, None
    
    def _stream_frames_worker(self, camera_id, width=0, height=0, fps=0, quality=90):
        """Worker function to stream frames from a camera."""
        try:
            request = camera_pb2.StreamRequest(
                camera_id=camera_id,
                width=width,
                height=height,
                fps=fps,
                quality=quality
            )
            
            retry_count = 0
            max_retries = 5
            retry_delay = 1.0
            
            while self.running and retry_count < max_retries:
                try:
                    logger.info(f"Starting stream for camera {camera_id}")
                    
                    for response in self.stub.StreamFrames(request):
                        if not self.running:
                            break
                        
                        if response.error:
                            logger.warning(f"Stream error for camera {camera_id}: {response.error}")
                            continue
                        
                        # Check if we have valid frame data
                        if not response.data:
                            logger.warning(f"Empty frame data for camera {camera_id}")
                            continue
                        
                        try:
                            # Decode JPEG frame
                            frame_data = np.frombuffer(response.data, dtype=np.uint8)
                            frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
                            
                            # Check if decode was successful
                            if frame is None:
                                logger.warning(f"Failed to decode frame for camera {camera_id}")
                                continue
                            
                            # Add frame to buffer
                            success = self.frame_buffer.put_frame(camera_id, frame, response.timestamp)
                            if not success:
                                logger.warning(f"Failed to buffer frame for camera {camera_id}")
                            
                            # Reset retry count on successful frame
                            retry_count = 0
                            
                        except Exception as e:
                            logger.error(f"Error processing frame: {str(e)}")
                    
                    # If we get here, the stream ended normally
                    logger.info(f"Stream ended for camera {camera_id}")
                    break
                    
                except grpc.RpcError as e:
                    retry_count += 1
                    logger.error(f"RPC error in stream for camera {camera_id} (try {retry_count}/{max_retries}): {e.code()}: {e.details()}")
                    if retry_count < max_retries:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        # Increase delay for next retry (exponential backoff)
                        retry_delay = min(retry_delay * 2, 10)
                    else:
                        logger.error(f"Maximum retry attempts reached for camera {camera_id}")
                        break
                        
        except Exception as e:
            logger.error(f"Unhandled error streaming frames for camera {camera_id}: {str(e)}")
    
    def start_streaming(self, camera_id, width=0, height=0, fps=0, quality=90):
        """Start streaming frames from the specified camera."""
        if camera_id in self.stream_threads and self.stream_threads[camera_id].is_alive():
            logger.warning(f"Stream already active for camera {camera_id}")
            return False
        
        logger.info(f"Starting stream for camera {camera_id}")
        self.running = True
        
        thread = threading.Thread(
            target=self._stream_frames_worker,
            args=(camera_id, width, height, fps, quality),
            daemon=True
        )
        self.stream_threads[camera_id] = thread
        thread.start()
        
        return True
    
    def start_streaming_multiple(self, camera_ids, width=0, height=0, fps=0, quality=90):
        """Start streaming from multiple cameras."""
        results = {}
        for camera_id in camera_ids:
            results[camera_id] = self.start_streaming(camera_id, width, height, fps, quality)
        return results
    
    def stop_streaming(self, camera_id=None):
        """Stop streaming frames."""
        self.running = False
        
        # Wait for threads to terminate
        if camera_id is not None:
            if camera_id in self.stream_threads:
                self.stream_threads[camera_id].join(timeout=2.0)
                logger.info(f"Stopped streaming for camera {camera_id}")
        else:
            for cam_id, thread in self.stream_threads.items():
                thread.join(timeout=2.0)
            logger.info("Stopped all streams")
    
    def get_latest_frame(self, camera_id):
        """Get the latest frame for the specified camera."""
        return self.frame_buffer.get_latest_frame(camera_id)
    
    def get_all_frames(self):
        """Get the latest frames for all cameras."""
        return self.frame_buffer.get_all_frames()
    
    def close(self):
        """Close the client connection."""
        self.stop_streaming()
        if self.channel is not None:
            self.channel.close()
            logger.info("Client connection closed")


def display_multiple_cameras(client, camera_ids, window_name="Camera Feed"):
    """Display feeds from multiple cameras in a grid layout."""
    # Initialize window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)  # Set initial window size
    
    # Start streaming from all cameras with a slight delay between each
    for camera_id in camera_ids:
        logger.info(f"Starting stream for camera {camera_id}")
        client.start_streaming(camera_id)
        time.sleep(0.5)  # Give each camera time to initialize
    
    # Counters for FPS calculation
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    # Counter for empty frame handling
    empty_frames_count = 0
    max_empty_frames = 50  # Maximum consecutive empty frames before warning
    
    try:
        while True:
            loop_start = time.time()
            
            # Get frames from all cameras
            frames = client.get_all_frames()
            
            if not frames:
                empty_frames_count += 1
                if empty_frames_count > max_empty_frames:
                    logger.warning(f"No frames available for {empty_frames_count} consecutive iterations")
                    empty_frames_count = 0  # Reset counter to avoid spamming logs
                    
                    # Try to restart streaming for all cameras
                    logger.info("Attempting to restart camera streams...")
                    client.stop_streaming()
                    time.sleep(1.0)
                    for camera_id in camera_ids:
                        client.start_streaming(camera_id)
                        time.sleep(0.5)
                
                time.sleep(0.1)
                continue
            else:
                empty_frames_count = 0  # Reset counter when frames are received
            
            # Update FPS calculation
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
            
            # Determine grid layout
            num_cameras = len(frames)
            grid_size = int(np.ceil(np.sqrt(num_cameras)))
            
            # Create blank grid with consistent dimensions
            cell_height = 240  # Fixed height for each camera feed
            cell_width = 320   # Fixed width for each camera feed
            grid_height = grid_size * cell_height
            grid_width = grid_size * cell_width
            grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
            
            # Log grid dimensions
            logger.debug(f"Grid dimensions: {grid_width}x{grid_height}, cell size: {cell_width}x{cell_height}")
            
            # Place frames in grid
            i = 0
            for camera_id, (frame, timestamp) in frames.items():
                if frame is None:
                    logger.warning(f"Null frame received for camera {camera_id}")
                    continue
                
                try:
                    # Calculate position in grid
                    row = i // grid_size
                    col = i % grid_size
                    
                    # Validate frame data
                    if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
                        logger.warning(f"Invalid frame dimensions for camera {camera_id}: {frame.shape}")
                        continue
                    
                    # Resize frame to fit in grid - ensure consistent dimensions
                    cell_width = 320   # Must match the grid cell width
                    cell_height = 240  # Must match the grid cell height
                    
                    try:
                        # Ensure frame is valid before resizing
                        if frame.shape[0] <= 0 or frame.shape[1] <= 0:
                            logger.warning(f"Invalid frame dimensions for camera {camera_id}: {frame.shape}")
                            continue
                            
                        frame_resized = cv2.resize(frame, (cell_width, cell_height))
                        
                        # Verify resize worked correctly
                        if frame_resized.shape != (cell_height, cell_width, 3):
                            logger.warning(f"Resize produced unexpected shape: {frame_resized.shape}, expected: ({cell_height}, {cell_width}, 3)")
                            # Force correct shape
                            frame_resized = cv2.resize(frame, (cell_width, cell_height), interpolation=cv2.INTER_NEAREST)
                    except Exception as e:
                        logger.error(f"Error resizing frame for camera {camera_id}: {e}")
                        continue
                    
                    # Add camera ID and timestamp
                    timestamp_str = time.strftime("%H:%M:%S", time.localtime(timestamp/1000)) if timestamp else "Unknown"
                    cv2.putText(frame_resized, f"Cam {camera_id} - {timestamp_str}", 
                                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Add overall FPS
                    cv2.putText(frame_resized, f"FPS: {fps:.1f}", 
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Place in grid - ensure dimensions match
                    try:
                        # Get target region dimensions
                        target_height = 240
                        target_width = 320  # Make sure this matches the resize dimensions above
                        
                        # Check grid boundaries
                        if (row+1)*target_height <= grid.shape[0] and (col+1)*target_width <= grid.shape[1]:
                            grid[row*target_height:(row+1)*target_height, 
                                col*target_width:(col+1)*target_width] = frame_resized
                        else:
                            logger.warning(f"Frame position outside grid boundaries: row={row}, col={col}")
                    except ValueError as e:
                        logger.error(f"Error placing frame in grid: {e}, frame shape={frame_resized.shape}, grid shape={grid.shape}")
                        # Additional failsafe approach
                        try:
                            # Ensure exact size match by forcing a resize to target dimensions
                            exact_frame = cv2.resize(frame_resized, (target_width, target_height))
                            grid[row*target_height:(row+1)*target_height, 
                                col*target_width:(col+1)*target_width] = exact_frame
                        except Exception as e2:
                            logger.error(f"Fallback resize also failed: {e2}")
                    
                    i += 1
                    
                except Exception as e:
                    logger.error(f"Error processing frame for camera {camera_id}: {e}")
            
            # Show the grid
            try:
                cv2.imshow(window_name, grid)
            except Exception as e:
                logger.error(f"Error displaying grid: {e}")
            
            # Check for key press to exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            # Control frame rate
            elapsed = time.time() - loop_start
            sleep_time = max(0, 1/30 - elapsed)  # Target 30 FPS
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    finally:
        # Clean up
        logger.info("Stopping streams and closing windows")
        client.stop_streaming()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Camera gRPC Client')
    parser.add_argument('--server', default='localhost:50051', help='Server address')
    parser.add_argument('--cameras', type=int, nargs='+', help='Camera IDs to stream')
    parser.add_argument('--list', action='store_true', help='List available cameras')
    args = parser.parse_args()
    
    # Create client
    client = CameraClient(server_address=args.server)
    
    try:
        if args.list:
            # List cameras
            cameras = client.list_cameras()
            print("Available cameras:")
            for camera in cameras:
                print(f"ID: {camera.id}, Name: {camera.name}, Resolution: {camera.width}x{camera.height}, FPS: {camera.fps}")
        elif args.cameras:
            # Stream from specified cameras
            display_multiple_cameras(client, args.cameras)
        else:
            # Default: try to stream from camera 0
            display_multiple_cameras(client, [0])
    
    finally:
        client.close()


if __name__ == '__main__':
    main()