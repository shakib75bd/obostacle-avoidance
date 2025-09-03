import cv2
import numpy as np
import time
from pathlib import Path
import threading
import queue

class VideoSource:
    def __init__(self, source=0, low_res_size=(320, 240), webcam_mode=False):
        """
        Initialize video source (webcam or video file)

        Args:
            source: 0 for webcam or path to video file
            low_res_size: Target resolution (width, height)
            webcam_mode: If True, optimize for real-time webcam processing
        """
        self.source = source
        self.low_res_size = low_res_size
        self.cap = None
        self.is_file = isinstance(source, (str, Path)) and Path(source).exists()
        self.buffer_size = 8  # Number of frames to buffer for smoother processing
        self.webcam_mode = webcam_mode

        # For timing control
        self.last_read_time = time.time()
        self.last_frame = None
        self.native_fps = 30  # Default, will be updated when opening the source
        self.frame_interval = 1.0 / self.native_fps  # Time between frames

    def __enter__(self):
        """Context manager entry"""
        print(f"Opening video source: {self.source}")
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video source {self.source}")

        # Get native fps for timing control
        self.native_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.native_fps <= 0 or self.native_fps > 120:  # Invalid or unrealistic fps
            self.native_fps = 30  # Default fallback

        self.frame_interval = 1.0 / self.native_fps

        # Print video information
        if self.is_file:
            frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            print(f"Video info: {frame_count} frames, {self.native_fps} fps")
            print(f"Frame interval: {self.frame_interval*1000:.1f} ms")
        else:
            # For webcam, try to get its properties
            webcam_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            webcam_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"Webcam native resolution: {webcam_width}x{webcam_height}")
            print(f"Webcam native fps: {self.native_fps}")
            print(f"Using resolution: {self.low_res_size[0]}x{self.low_res_size[1]}")

        # Set resolution for webcam (won't affect video files)
        if not self.is_file:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.low_res_size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.low_res_size[1])

        # For video files, set buffer size for better throughput
        if self.is_file:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.cap:
            self.cap.release()

    def read(self):
        """Read a frame from the video source"""
        if not self.cap or not self.cap.isOpened():
            return False, None

        # Handle frame timing based on source type
        current_time = time.time()
        elapsed = current_time - self.last_read_time

        if self.is_file:
            # For video files, we want to maintain the original video's timing
            # if we're processing faster than real-time
            if not self.webcam_mode and elapsed < self.frame_interval:
                # We're processing too fast, return the previous frame
                # to maintain the original video's fps
                if self.last_frame is not None:
                    # Artificial delay to match original video timing
                    # Only add small delay to avoid blocking UI
                    sleep_time = min(0.001, self.frame_interval - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    return True, self.last_frame
        else:
            # For webcam in real-time mode
            if self.webcam_mode:
                # Always process at native webcam fps
                # If we're processing faster than the webcam can provide new frames,
                # just get the latest one
                pass
            else:
                # For webcam in normal mode, maintain steady fps
                if elapsed < self.frame_interval:
                    # We're processing too fast
                    if self.last_frame is not None:
                        return True, self.last_frame

        # Read a new frame
        self.last_read_time = current_time
        ret, frame = self.cap.read()
        if not ret:
            return False, None

        # Resize if not already at target resolution
        if frame.shape[1] != self.low_res_size[0] or frame.shape[0] != self.low_res_size[1]:
            frame = cv2.resize(frame, self.low_res_size, interpolation=cv2.INTER_AREA)

        # Store the frame for potential reuse
        self.last_frame = frame.copy()

        return True, frame

    def get_fps(self):
        """Get the FPS of the video source"""
        if self.cap:
            return self.cap.get(cv2.CAP_PROP_FPS)
        return 30  # Default assumption

    def get_frame_count(self):
        """Get total frame count (only works for video files)"""
        if self.cap and self.is_file:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return -1  # Infinite for webcam


class FPSCounter:
    """Simple FPS counter for performance measurement"""
    def __init__(self, window_size=30):
        self.prev_time = time.time()
        self.frame_times = []
        self.window_size = window_size

    def update(self):
        curr_time = time.time()
        dt = curr_time - self.prev_time
        self.prev_time = curr_time

        self.frame_times.append(dt)
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)

    def get_fps(self):
        if not self.frame_times:
            return 0
        return len(self.frame_times) / sum(self.frame_times)
