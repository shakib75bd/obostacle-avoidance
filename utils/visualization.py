import cv2
import numpy as np
import time
from collections import deque

class PerformanceMetrics:
    """Tracks performance metrics for evaluation"""
    def __init__(self, window_size=100):
        self.metrics = {
            'depth_time': deque(maxlen=window_size),
            'detection_time': deque(maxlen=window_size),
            'fusion_time': deque(maxlen=window_size),
            'total_time': deque(maxlen=window_size),
            'fps': deque(maxlen=window_size),
            'skip_rate': deque(maxlen=window_size)
        }

    def update(self, metric_name, value):
        """Add a new measurement"""
        if metric_name in self.metrics:
            self.metrics[metric_name].append(value)

    def get_average(self, metric_name):
        """Get average of a metric"""
        if metric_name in self.metrics and self.metrics[metric_name]:
            return sum(self.metrics[metric_name]) / len(self.metrics[metric_name])
        return 0

    def get_summary(self):
        """Get summary of all metrics"""
        return {
            metric: self.get_average(metric)
            for metric in self.metrics
        }


def create_visualization(original_img, depth_colored, uncertainty_colored,
                        detection_img, obstacle_viz, fps, metrics=None, webcam_mode=False):
    """
    Create a composite visualization of all components

    Args:
        original_img: Original input image
        depth_colored: Colored depth map
        uncertainty_colored: Colored uncertainty map
        detection_img: Image with detection boxes
        obstacle_viz: Obstacle map visualization
        fps: Current FPS
        metrics: Performance metrics
        webcam_mode: If True, show webcam mode indicator

    Returns:
        visualization: Composite visualization image
    """
    h, w = original_img.shape[:2]

    # Lower resolution for visualization to improve performance
    viz_scale = 0.5
    viz_w, viz_h = int(w * viz_scale), int(h * viz_scale)

    # Resize all images to smaller size for faster visualization
    orig_small = cv2.resize(original_img, (viz_w, viz_h), interpolation=cv2.INTER_AREA)
    depth_small = cv2.resize(depth_colored, (viz_w, viz_h), interpolation=cv2.INTER_AREA)
    uncert_small = cv2.resize(uncertainty_colored, (viz_w, viz_h), interpolation=cv2.INTER_AREA)
    detect_small = cv2.resize(detection_img, (viz_w, viz_h), interpolation=cv2.INTER_AREA)
    obstacle_small = cv2.resize(obstacle_viz, (viz_w, viz_h), interpolation=cv2.INTER_AREA)

    # Create top row and bottom row
    top_row = np.hstack([orig_small, detect_small])
    bottom_row = np.hstack([depth_small, obstacle_small])

    # Stack rows
    visualization = np.vstack([top_row, bottom_row])

    # Resize back to original size for display
    visualization = cv2.resize(visualization, (w*2, h*2), interpolation=cv2.INTER_LINEAR)

    # Add performance information
    cv2.putText(
        visualization, f"Processing FPS: {fps:.1f}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
    )

    # Add status label showing we're operating at standard 24fps
    cv2.putText(
        visualization, f"Output: 24fps Standard", (w, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
    )

    # Add webcam mode indicator if active
    if webcam_mode:
        cv2.putText(
            visualization, "WEBCAM REAL-TIME MODE", (w, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2
        )

    # Add skipping information
    if metrics and 'skip_rate' in metrics.metrics and metrics.metrics['skip_rate']:
        skip_rate = metrics.get_average('skip_rate')
        cv2.putText(
            visualization, f"Frame Skip: {int(skip_rate)} frames", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2
        )

    if metrics:
        summary = metrics.get_summary()
        y_pos = 90  # Moved down to accommodate the skip rate display
        for metric, value in summary.items():
            if metric == 'fps' or metric == 'skip_rate':
                continue  # Already displayed above

            # Convert time to ms
            if 'time' in metric:
                text = f"{metric}: {value*1000:.1f} ms"
            else:
                text = f"{metric}: {value:.2f}"

            cv2.putText(
                visualization, text, (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1
            )
            y_pos += 30

    # Add borders between images
    # Horizontal line
    cv2.line(
        visualization,
        (0, h),
        (w*2, h),
        (255, 255, 255),
        2
    )

    # Vertical lines
    cv2.line(
        visualization,
        (w, 0),
        (w, h*2),
        (255, 255, 255),
        2
    )

    return visualization
