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

        # Cache for obstacle maps
        self.obstacle_cache = {}
        self.cache_size = 5

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
        # Create a cache key based on inputs
        cache_key = (
            hash(depth_map.tobytes()),
            hash(uncertainty_map.tobytes()),
            hash(str(detections))  # Simple hash of detections
        )

        # Check cache first
        if cache_key in self.obstacle_cache:
            return self.obstacle_cache[cache_key]

        h, w = img_shape[:2]

        # Use lower resolution for processing to speed up
        scale_factor = 0.5
        process_h, process_w = int(h * scale_factor), int(w * scale_factor)

        # Resize inputs for faster processing
        depth_small = cv2.resize(depth_map, (process_w, process_h), interpolation=cv2.INTER_AREA)
        uncertainty_small = cv2.resize(uncertainty_map, (process_w, process_h), interpolation=cv2.INTER_AREA)

        # Initialize obstacle map
        obstacle_map = np.zeros((process_h, process_w), dtype=np.float32)

        # Create binary masks for confidence regions
        high_confidence = uncertainty_small < self.uncertainty_threshold
        low_confidence = ~high_confidence

        # 1. In high confidence regions, use depth map
        # Objects closer to camera (smaller depth) are more likely to be obstacles
        # We invert the depth map so closer objects have higher values
        depth_obstacle_likelihood = 1.0 - depth_small.copy()

        # Only consider depths in a certain range as obstacles
        # Very close (could be noise) and very far objects are less relevant
        depth_obstacle_likelihood[(depth_small < self.min_obstacle_depth) |
                                 (depth_small > self.max_obstacle_depth)] = 0

        # Apply to high confidence regions
        obstacle_map[high_confidence] = depth_obstacle_likelihood[high_confidence]

        # 2. In low confidence regions, rely more on object detections
        # Create mask from detections
        detection_mask = np.zeros((process_h, process_w), dtype=np.float32)

        # Scale detection boxes for the smaller processing size
        for det in detections:
            x1, y1, x2, y2 = det['box']

            # Scale box coordinates to smaller size
            x1 = int(x1 * scale_factor)
            y1 = int(y1 * scale_factor)
            x2 = int(x2 * scale_factor)
            y2 = int(y2 * scale_factor)

            # Dilate box slightly
            box_width, box_height = x2 - x1, y2 - y1
            dilation_x = int(box_width * self.box_dilation_factor)
            dilation_y = int(box_height * self.box_dilation_factor)

            # Ensure coordinates are within image bounds
            x1 = max(0, x1 - dilation_x)
            y1 = max(0, y1 - dilation_y)
            x2 = min(process_w, x2 + dilation_x)
            y2 = min(process_h, y2 + dilation_y)

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

        # Apply smaller kernel for speed
        obstacle_map = cv2.GaussianBlur(obstacle_map, (5, 5), 1.0)

        # Resize back to original size
        obstacle_map_full = cv2.resize(obstacle_map, (w, h), interpolation=cv2.INTER_LINEAR)

        # Create visualization - colored heatmap
        colored_map = cv2.applyColorMap(
            (obstacle_map_full * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )

        # Cache the result
        result = (obstacle_map_full, colored_map)
        self.obstacle_cache[cache_key] = result

        # Maintain cache size
        if len(self.obstacle_cache) > self.cache_size:
            # Remove oldest item
            self.obstacle_cache.pop(next(iter(self.obstacle_cache)))

        return result

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

    def determine_navigation_direction(self, obstacle_map, threshold=0.5):
        """
        Determine if it's safe to move forward or if turning is required based on obstacle density

        Args:
            obstacle_map: Obstacle likelihood map (0-1)
            threshold: Threshold for obstacle detection

        Returns:
            navigation_type: 0 for MOVE FORWARD, 1 for TURN
            obstacle_density: The density of obstacles in front (0-1)
        """
        h, w = obstacle_map.shape

        # Define the front region (bottom center of the image)
        # This is the critical area for forward movement
        front_y1 = int(h * 0.6)  # Bottom 40% of the image
        front_y2 = h
        front_x1 = int(w * 0.3)  # Center 40% of the image
        front_x2 = int(w * 0.7)

        # Extract the front region
        front_region = obstacle_map[front_y1:front_y2, front_x1:front_x2]

        # Calculate obstacle density in front region
        obstacle_density = np.mean(front_region)

        # Determine if forward movement is safe
        # If obstacle density is below threshold, suggest moving forward
        # Otherwise, suggest turning
        if obstacle_density < threshold:
            navigation_type = 0  # MOVE FORWARD
        else:
            navigation_type = 1  # TURN

        return navigation_type, obstacle_density
