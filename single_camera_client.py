#!/usr/bin/env python3
# single_camera_viewer.py - A focused utility for streaming a single camera

import cv2
import argparse
import time
import logging
import sys
import signal
import threading
from datetime import datetime
import os
import numpy as np

# Import the camera client
from camera_client import CameraClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SingleCameraViewer:
    """A focused viewer for a single camera stream."""
    
    def __init__(self, server_address, camera_id, width=0, height=0, fps=0, quality=90):
        """Initialize the viewer with camera settings."""
        self.server_address = server_address
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.quality = quality
        
        self.client = None
        self.running = False
        self.recording = False
        self.video_writer = None
        self.statistics = {
            'frames_received': 0,
            'frames_displayed': 0,
            'start_time': None,
            'fps': 0,
            'display_lag': 0
        }
        self.last_frame_time = 0
        
        # Initialize the output directory for recordings
        self.output_dir = "recordings"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Track frame dimensions for consistent recording
        self.frame_dimensions = None
    
    def connect(self):
        """Connect to the gRPC server."""
        try:
            self.client = CameraClient(server_address=self.server_address)
            logger.info(f"Connected to server at {self.server_address}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to server: {str(e)}")
            return False
    
    def start_streaming(self):
        """Start streaming from the camera."""
        if not self.client:
            if not self.connect():
                return False
        
        # Get camera info if possible
        try:
            info = self.client.get_camera_info(self.camera_id)
            if info:
                logger.info(f"Camera info: {info.name}, {info.width}x{info.height} @ {info.fps}fps")
                
                # Use camera's native resolution if no custom resolution specified
                if self.width == 0:
                    self.width = info.width
                if self.height == 0:
                    self.height = info.height
                if self.fps == 0:
                    self.fps = info.fps
        except Exception as e:
            logger.warning(f"Could not get camera info: {str(e)}")
        
        # Start streaming
        success = self.client.start_streaming(
            self.camera_id,
            width=self.width,
            height=self.height,
            fps=self.fps,
            quality=self.quality
        )
        
        if success:
            logger.info(f"Started streaming camera {self.camera_id}")
            self.running = True
            self.statistics['start_time'] = time.time()
            return True
        else:
            logger.error(f"Failed to start streaming camera {self.camera_id}")
            return False
    
    def start_recording(self):
        """Start recording the camera stream to a file."""
        if self.recording:
            logger.warning("Already recording")
            return False
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"camera_{self.camera_id}_{timestamp}.mp4")
        
        # We need a frame to determine dimensions
        if not self.frame_dimensions:
            logger.info("Waiting for first frame to determine dimensions...")
            for _ in range(30):  # Wait for up to 3 seconds
                frame, _ = self.client.get_latest_frame(self.camera_id)
                if frame is not None:
                    self.frame_dimensions = (frame.shape[1], frame.shape[0])
                    break
                time.sleep(0.1)
        
        if not self.frame_dimensions:
            logger.error("Could not determine frame dimensions for recording")
            return False
        
        # Create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        try:
            self.video_writer = cv2.VideoWriter(
                filename, 
                fourcc, 
                self.fps if self.fps > 0 else 30, 
                self.frame_dimensions
            )
            
            if not self.video_writer.isOpened():
                logger.error(f"Failed to create VideoWriter for {filename}")
                self.video_writer = None
                return False
            
            self.recording = True
            logger.info(f"Started recording to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error starting recording: {str(e)}")
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            return False
    
    def stop_recording(self):
        """Stop the current recording."""
        if not self.recording:
            return
        
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        self.recording = False
        logger.info("Recording stopped")
    
    def update_statistics(self, frame_time):
        """Update streaming statistics."""
        self.statistics['frames_received'] += 1
        
        # Calculate FPS over a window
        elapsed = time.time() - self.statistics['start_time']
        if elapsed > 0:
            self.statistics['fps'] = self.statistics['frames_received'] / elapsed
        
        # Calculate display lag
        if frame_time:
            current_time = time.time() * 1000  # Convert to ms
            self.statistics['display_lag'] = current_time - frame_time
    
    def display(self):
        """Display the camera stream in a window."""
        if not self.running:
            if not self.start_streaming():
                logger.error("Failed to start streaming")
                return
        
        # Create window
        window_name = f"Camera {self.camera_id}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        
        # Start time for FPS calculation
        last_fps_update = time.time()
        fps_display = 0
        frames_since_update = 0
        
        try:
            while self.running:
                # Get the latest frame
                frame, timestamp = self.client.get_latest_frame(self.camera_id)
                
                if frame is not None:
                    # Save the frame dimensions for recording
                    if not self.frame_dimensions:
                        self.frame_dimensions = (frame.shape[1], frame.shape[0])
                    
                    # Update statistics
                    self.update_statistics(timestamp)
                    frames_since_update += 1
                    
                    # Record frame if recording is active
                    if self.recording and self.video_writer:
                        try:
                            self.video_writer.write(frame)
                        except Exception as e:
                            logger.error(f"Error writing frame to video: {str(e)}")
                    
                    # Update FPS display every second
                    current_time = time.time()
                    if current_time - last_fps_update >= 1.0:
                        fps_display = frames_since_update / (current_time - last_fps_update)
                        frames_since_update = 0
                        last_fps_update = current_time
                    
                    # Add information overlay
                    frame_with_info = frame.copy()
                    
                    # Add camera info
                    cv2.putText(frame_with_info, f"Camera {self.camera_id}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Add FPS info
                    cv2.putText(frame_with_info, f"FPS: {fps_display:.1f}", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Add timestamp if available
                    if timestamp:
                        timestamp_str = datetime.fromtimestamp(timestamp/1000).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                        cv2.putText(frame_with_info, timestamp_str, 
                                   (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Add lag information
                    cv2.putText(frame_with_info, f"Lag: {self.statistics['display_lag']:.1f}ms", 
                               (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Add recording indicator
                    if self.recording:
                        # Red circle and REC text
                        cv2.circle(frame_with_info, (frame.shape[1] - 30, 30), 15, (0, 0, 255), -1)
                        cv2.putText(frame_with_info, "REC", 
                                   (frame.shape[1] - 80, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # Display the frame
                    cv2.imshow(window_name, frame_with_info)
                    self.statistics['frames_displayed'] += 1
                else:
                    # If no frame is available, display a black screen with info
                    if self.frame_dimensions:
                        width, height = self.frame_dimensions
                    else:
                        width, height = 640, 480
                    
                    blank = np.zeros((height, width, 3), dtype=np.uint8)
                    cv2.putText(blank, f"Waiting for camera {self.camera_id}...", 
                               (width//2 - 150, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.imshow(window_name, blank)
                
                # Process keyboard input
                key = cv2.waitKey(1) & 0xFF
                self.process_key(key)
                
                # Don't burn CPU if no frame
                if frame is None:
                    time.sleep(0.1)
            
        except KeyboardInterrupt:
            logger.info("Display interrupted by user")
        finally:
            self.stop_recording()
            cv2.destroyAllWindows()
            self.client.stop_streaming(self.camera_id)
            self.client.close()
            logger.info("Display stopped")
    
    def process_key(self, key):
        """Process keyboard input."""
        if key == ord('q'):  # Quit
            self.running = False
        elif key == ord('r'):  # Toggle recording
            if self.recording:
                self.stop_recording()
            else:
                self.start_recording()
        elif key == ord('s'):  # Take screenshot
            frame, _ = self.client.get_latest_frame(self.camera_id)
            if frame is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_camera_{self.camera_id}_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                logger.info(f"Screenshot saved to {filename}")
        elif key == ord('i'):  # Show camera info
            try:
                info = self.client.get_camera_info(self.camera_id)
                if info:
                    logger.info(f"Camera info: {info.name}, {info.width}x{info.height} @ {info.fps}fps")
            except Exception as e:
                logger.error(f"Error getting camera info: {str(e)}")
        elif key == ord('f'):  # Toggle fullscreen
            window_name = f"Camera {self.camera_id}"
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN) != cv2.WINDOW_FULLSCREEN:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Single Camera Viewer')
    parser.add_argument('--server', default='localhost:50051', help='Server address')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID to stream')
    parser.add_argument('--width', type=int, default=0, help='Requested width (0 for default)')
    parser.add_argument('--height', type=int, default=0, help='Requested height (0 for default)')
    parser.add_argument('--fps', type=int, default=0, help='Requested FPS (0 for default)')
    parser.add_argument('--quality', type=int, default=90, help='JPEG quality (1-100)')
    args = parser.parse_args()
    
    # Create viewer
    viewer = SingleCameraViewer(
        server_address=args.server,
        camera_id=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        quality=args.quality
    )
    
    # Set up signal handler for clean shutdown
    def signal_handler(sig, frame):
        logger.info("Received interrupt signal, shutting down")
        viewer.running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start display
    viewer.display()

if __name__ == '__main__':
    main()