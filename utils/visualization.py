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
            'fps': deque(maxlen=window_size)
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
                        detection_img, obstacle_viz, fps, metrics=None):
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

    Returns:
        visualization: Composite visualization image
    """
    h, w = original_img.shape[:2]

    # Resize all images to same size
    depth_colored = cv2.resize(depth_colored, (w, h))
    uncertainty_colored = cv2.resize(uncertainty_colored, (w, h))
    detection_img = cv2.resize(detection_img, (w, h))
    obstacle_viz = cv2.resize(obstacle_viz, (w, h))

    # Create top row and bottom row
    top_row = np.hstack([original_img, detection_img])
    bottom_row = np.hstack([depth_colored, obstacle_viz])

    # Stack rows
    visualization = np.vstack([top_row, bottom_row])

    # Add performance information
    cv2.putText(
        visualization, f"FPS: {fps:.1f}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )

    if metrics:
        summary = metrics.get_summary()
        y_pos = 70
        for metric, value in summary.items():
            if metric == 'fps':
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
