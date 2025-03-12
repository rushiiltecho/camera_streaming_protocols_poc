# multi_camera_viewer.py
import cv2
import numpy as np
import argparse
import threading
import time
import os
from datetime import datetime
import logging
from camera_client import CameraClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CameraRecorder:
    """Class to handle recording of camera feeds."""
    
    def __init__(self, output_dir="recordings"):
        self.output_dir = output_dir
        self.recording = {}  # Dictionary mapping camera_id to recording state
        self.writers = {}    # Dictionary mapping camera_id to VideoWriter
        self.lock = threading.Lock()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def start_recording(self, camera_id, frame_size, fps=30):
        """Start recording for the specified camera."""
        with self.lock:
            if camera_id in self.recording and self.recording[camera_id]:
                logger.warning(f"Recording already active for camera {camera_id}")
                return False
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"camera_{camera_id}_{timestamp}.mp4")
            
            # Create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
            
            if not writer.isOpened():
                logger.error(f"Failed to create VideoWriter for camera {camera_id}")
                return False
            
            self.writers[camera_id] = writer
            self.recording[camera_id] = True
            
            logger.info(f"Started recording for camera {camera_id} to {filename}")
            return True
    
    def stop_recording(self, camera_id):
        """Stop recording for the specified camera."""
        with self.lock:
            if camera_id not in self.recording or not self.recording[camera_id]:
                logger.warning(f"No active recording for camera {camera_id}")
                return False
            
            # Release VideoWriter
            if camera_id in self.writers:
                self.writers[camera_id].release()
                del self.writers[camera_id]
            
            self.recording[camera_id] = False
            
            logger.info(f"Stopped recording for camera {camera_id}")
            return True
    
    def record_frame(self, camera_id, frame):
        """Record a frame for the specified camera."""
        with self.lock:
            if camera_id not in self.recording or not self.recording[camera_id]:
                return False
            
            if camera_id in self.writers:
                self.writers[camera_id].write(frame)
                return True
            
            return False
    
    def is_recording(self, camera_id):
        """Check if the specified camera is currently recording."""
        with self.lock:
            return camera_id in self.recording and self.recording[camera_id]
    
    def stop_all_recordings(self):
        """Stop all active recordings."""
        with self.lock:
            camera_ids = list(self.recording.keys())
        
        for camera_id in camera_ids:
            self.stop_recording(camera_id)


class MultiCameraViewer:
    """Class to display and control multiple camera feeds."""
    
    def __init__(self, server_address='localhost:50051'):
        self.client = CameraClient(server_address)
        self.recorder = CameraRecorder()
        self.selected_camera = None
        self.fullscreen_mode = False
        self.run_analytics = False
        self.show_info = True
    
    def list_cameras(self):
        """List all available cameras."""
        cameras = self.client.list_cameras()
        logger.info(f"Found {len(cameras)} cameras:")
        for camera in cameras:
            logger.info(f"ID: {camera.id}, Name: {camera.name}, "
                        f"Resolution: {camera.width}x{camera.height}, FPS: {camera.fps}")
        return cameras
    
    def display_grid(self, camera_ids, window_name="Multi-Camera Viewer"):
        """Display camera feeds in a grid layout with controls."""
        # Initialize window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Set initial window size
        cv2.resizeWindow(window_name, 1280, 720)
        
        # Start streaming from all cameras
        self.client.start_streaming_multiple(camera_ids)
        
        # Track fps
        fps_counters = {camera_id: 0 for camera_id in camera_ids}
        fps_times = {camera_id: time.time() for camera_id in camera_ids}
        fps_values = {camera_id: 0 for camera_id in camera_ids}
        
        try:
            while True:
                start_time = time.time()
                
                # Get frames from all cameras
                all_frames = self.client.get_all_frames()
                
                if not all_frames:
                    time.sleep(0.01)
                    continue
                
                # Handle fullscreen mode for selected camera
                if self.fullscreen_mode and self.selected_camera is not None:
                    if self.selected_camera in all_frames:
                        frame, timestamp = all_frames[self.selected_camera]
                        if frame is not None:
                            # Update FPS counter
                            fps_counters[self.selected_camera] += 1
                            if time.time() - fps_times[self.selected_camera] >= 1.0:
                                fps_values[self.selected_camera] = fps_counters[self.selected_camera]
                                fps_counters[self.selected_camera] = 0
                                fps_times[self.selected_camera] = time.time()
                            
                            # Record frame if recording is active
                            if self.recorder.is_recording(self.selected_camera):
                                self.recorder.record_frame(self.selected_camera, frame)
                            
                            # Run analytics if enabled
                            if self.run_analytics:
                                frame = self.apply_analytics(frame)
                            
                            # Add overlay information
                            if self.show_info:
                                self.add_overlay(frame, self.selected_camera, timestamp, 
                                                fps_values[self.selected_camera])
                            
                            # Display frame
                            cv2.imshow(window_name, frame)
                    else:
                        self.fullscreen_mode = False
                else:
                    # Determine grid layout
                    num_cameras = len(all_frames)
                    grid_size = int(np.ceil(np.sqrt(num_cameras)))
                    
                    # Create blank grid
                    grid_height = grid_size * 270  # Add some padding for info
                    grid_width = grid_size * 360
                    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
                    
                    # Place frames in grid
                    i = 0
                    for camera_id, (frame, timestamp) in all_frames.items():
                        if frame is None:
                            continue
                        
                        # Update FPS counter
                        fps_counters[camera_id] += 1
                        if time.time() - fps_times[camera_id] >= 1.0:
                            fps_values[camera_id] = fps_counters[camera_id]
                            fps_counters[camera_id] = 0
                            fps_times[camera_id] = time.time()
                        
                        # Record frame if recording is active
                        if self.recorder.is_recording(camera_id):
                            self.recorder.record_frame(camera_id, frame)
                        
                        # Run analytics if enabled
                        if self.run_analytics:
                            frame = self.apply_analytics(frame)
                        
                        # Calculate position in grid
                        row = i // grid_size
                        col = i % grid_size
                        
                        # Resize frame to fit in grid
                        frame_resized = cv2.resize(frame, (360, 270))
                        
                        # Add overlay information
                        if self.show_info:
                            self.add_overlay(frame_resized, camera_id, timestamp, 
                                            fps_values[camera_id])
                        
                        # Highlight selected camera
                        if camera_id == self.selected_camera:
                            cv2.rectangle(frame_resized, (0, 0), (360, 270), (0, 255, 0), 2)
                        
                        # Place in grid
                        grid[row*270:(row+1)*270, col*360:(col+1)*360] = frame_resized
                        i += 1
                    
                    # Display grid
                    cv2.imshow(window_name, grid)
                
                # Process keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):  # Quit
                    break
                elif key == ord('f'):  # Toggle fullscreen
                    self.fullscreen_mode = not self.fullscreen_mode
                elif key == ord('i'):  # Toggle info overlay
                    self.show_info = not self.show_info
                elif key == ord('a'):  # Toggle analytics
                    self.run_analytics = not self.run_analytics
                elif key == ord('r'):  # Toggle recording for selected camera
                    if self.selected_camera is not None:
                        if self.recorder.is_recording(self.selected_camera):
                            self.recorder.stop_recording(self.selected_camera)
                        else:
                            # Get frame size
                            if self.selected_camera in all_frames:
                                frame, _ = all_frames[self.selected_camera]
                                if frame is not None:
                                    height, width = frame.shape[:2]
                                    self.recorder.start_recording(self.selected_camera, (width, height))
                elif key == ord('s'):  # Take screenshot of selected camera
                    if self.selected_camera is not None and self.selected_camera in all_frames:
                        frame, _ = all_frames[self.selected_camera]
                        if frame is not None:
                            # Create screenshots directory if it doesn't exist
                            os.makedirs("screenshots", exist_ok=True)
                            # Save screenshot
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"screenshots/camera_{self.selected_camera}_{timestamp}.jpg"
                            cv2.imwrite(filename, frame)
                            logger.info(f"Screenshot saved: {filename}")
                elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), 
                            ord('6'), ord('7'), ord('8'), ord('9')]:
                    # Select camera by number key
                    camera_idx = key - ord('1')
                    if camera_idx < len(camera_ids):
                        self.selected_camera = camera_ids[camera_idx]
                        logger.info(f"Selected camera {self.selected_camera}")
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Limit to 30 FPS
                if processing_time < 1.0/30.0:
                    time.sleep(1.0/30.0 - processing_time)
        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        
        finally:
            # Clean up
            self.recorder.stop_all_recordings()
            self.client.stop_streaming()
            cv2.destroyAllWindows()
    
    def add_overlay(self, frame, camera_id, timestamp, fps):
        """Add information overlay to frame."""
        # Convert timestamp to readable format
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", 
                                     time.localtime(timestamp/1000)) if timestamp else "Unknown"
        
        # Add camera info
        cv2.putText(frame, f"Camera {camera_id}", 
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Add timestamp
        cv2.putText(frame, timestamp_str, 
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Add FPS
        cv2.putText(frame, f"FPS: {fps}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Add recording indicator
        if self.recorder.is_recording(camera_id):
            cv2.putText(frame, "REC", 
                       (frame.shape[1] - 50, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 0, 255), 1)
            # Add red circle indicator
            cv2.circle(frame, (frame.shape[1] - 20, 15), 8, (0, 0, 255), -1)
    
    def apply_analytics(self, frame):
        """Apply simple analytics to frame (motion detection)."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Store first frame for comparison
        if not hasattr(self, 'first_frame'):
            self.first_frame = blur
            return frame
        
        # Compute absolute difference between current frame and first frame
        frame_delta = cv2.absdiff(self.first_frame, blur)
        
        # Apply threshold
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        
        # Dilate thresholded image to fill in holes
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours on original frame
        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) < 500:  # Filter small contours
                continue
            
            motion_detected = True
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Update first frame for next iteration (weighted average)
        alpha = 0.1
        self.first_frame = cv2.addWeighted(blur, alpha, self.first_frame, 1.0 - alpha, 0)
        
        # Add motion detection indicator
        if motion_detected:
            cv2.putText(frame, "Motion Detected", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 0, 255), 2)
        
        return frame
    
    def close(self):
        """Close the viewer and release resources."""
        self.recorder.stop_all_recordings()
        self.client.close()


def main():
    parser = argparse.ArgumentParser(description='Multi-Camera Viewer')
    parser.add_argument('--server', default='localhost:50051', help='Server address')
    parser.add_argument('--cameras', type=int, nargs='+', help='Camera IDs to view')
    args = parser.parse_args()
    
    # Create viewer
    viewer = MultiCameraViewer(server_address=args.server)
    
    try:
        # If no cameras specified, list available cameras
        if not args.cameras:
            cameras = viewer.list_cameras()
            if cameras:
                camera_ids = [camera.id for camera in cameras]
            else:
                camera_ids = [0]  # Default to camera 0
        else:
            camera_ids = args.cameras
        
        # Display camera grid
        viewer.display_grid(camera_ids)
    
    finally:
        viewer.close()


if __name__ == '__main__':
    main()