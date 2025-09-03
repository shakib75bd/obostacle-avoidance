import cv2
import numpy as np
import torch
import torch.nn.functional as F

class ObstacleMapGenerator:
    """
    Generates obstacle likelihood maps by fusing depth, uncertainty, and object detection
    """
    def __init__(self,
                 uncertainty_threshold=0.3,
                 min_obstacle_depth=0.4,
                 max_obstacle_depth=0.8,
                 box_dilation_factor=0.1):
        """
        Initialize obstacle map generator

        Args:
            uncertainty_threshold: Threshold for high/low confidence regions
            min_obstacle_depth: Minimum normalized depth to consider as obstacle
            max_obstacle_depth: Maximum normalized depth to consider as obstacle
            box_dilation_factor: Factor to dilate detection boxes
        """
        self.uncertainty_threshold = uncertainty_threshold
        self.min_obstacle_depth = min_obstacle_depth
        self.max_obstacle_depth = max_obstacle_depth
        self.box_dilation_factor = box_dilation_factor

    def generate_obstacle_map(self, depth_map, uncertainty_map, detections, img_shape):
        """
        Generate obstacle likelihood map by fusing depth and detections

        Args:
            depth_map: Normalized depth map (0-1)
            uncertainty_map: Uncertainty map (0-1)
            detections: List of obstacle detections from object detector
            img_shape: Shape of the input image (h, w)

        Returns:
            obstacle_map: Obstacle likelihood map (0-1)
            visualization: Visualization of the obstacle map
        """
        h, w = img_shape[:2]

        # Initialize obstacle map
        obstacle_map = np.zeros((h, w), dtype=np.float32)

        # Create binary masks for confidence regions
        high_confidence = uncertainty_map < self.uncertainty_threshold
        low_confidence = ~high_confidence

        # 1. In high confidence regions, use depth map
        # Objects closer to camera (smaller depth) are more likely to be obstacles
        # We invert the depth map so closer objects have higher values
        depth_obstacle_likelihood = 1.0 - depth_map.copy()

        # Only consider depths in a certain range as obstacles
        # Very close (could be noise) and very far objects are less relevant
        depth_obstacle_likelihood[(depth_map < self.min_obstacle_depth) |
                                 (depth_map > self.max_obstacle_depth)] = 0

        # Apply to high confidence regions
        obstacle_map[high_confidence] = depth_obstacle_likelihood[high_confidence]

        # 2. In low confidence regions, rely more on object detections
        # Create mask from detections
        detection_mask = np.zeros((h, w), dtype=np.float32)

        for det in detections:
            x1, y1, x2, y2 = det['box']

            # Dilate box slightly
            box_width, box_height = x2 - x1, y2 - y1
            dilation_x = int(box_width * self.box_dilation_factor)
            dilation_y = int(box_height * self.box_dilation_factor)

            # Ensure coordinates are within image bounds
            x1 = max(0, x1 - dilation_x)
            y1 = max(0, y1 - dilation_y)
            x2 = min(w, x2 + dilation_x)
            y2 = min(h, y2 + dilation_y)

            # Weight by confidence
            confidence = det['conf']
            detection_mask[y1:y2, x1:x2] = max(detection_mask[y1:y2, x1:x2].max(), confidence)

        # Apply detection mask in low confidence regions
        obstacle_map[low_confidence] = np.maximum(
            obstacle_map[low_confidence],
            detection_mask[low_confidence]
        )

        # 3. Normalize and smooth the obstacle map
        if obstacle_map.max() > 0:
            obstacle_map = obstacle_map / obstacle_map.max()

        # Apply Gaussian blur for smoothing
        obstacle_map = cv2.GaussianBlur(obstacle_map, (7, 7), 1.5)

        # Create visualization
        visualization = cv2.applyColorMap(
            (obstacle_map * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )

        # Add confidence region overlay
        confidence_overlay = np.zeros((h, w, 3), dtype=np.uint8)
        confidence_overlay[low_confidence] = [0, 0, 128]  # Dark red for low confidence

        # Blend with 30% opacity
        visualization = cv2.addWeighted(
            visualization, 1.0,
            confidence_overlay, 0.3,
            0
        )

        return obstacle_map, visualization

    def extract_obstacle_regions(self, obstacle_map, threshold=0.5):
        """
        Extract obstacle regions from the obstacle map

        Args:
            obstacle_map: Obstacle likelihood map (0-1)
            threshold: Threshold for obstacle detection

        Returns:
            obstacle_regions: Binary mask of obstacle regions
            stats: Stats about each connected obstacle region
        """
        # Threshold obstacle map
        binary_obstacles = (obstacle_map > threshold).astype(np.uint8) * 255

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_obstacles, connectivity=8
        )

        # Skip background (label 0)
        obstacle_regions = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            cx, cy = centroids[i]

            if area > 50:  # Filter very small regions
                obstacle_regions.append({
                    'label': i,
                    'box': (x, y, x+w, y+h),
                    'centroid': (cx, cy),
                    'area': area
                })

        return binary_obstacles, obstacle_regions
