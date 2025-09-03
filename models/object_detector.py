import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time

class ObjectDetector:
    """
    YOLOv8 object detection module
    """
    def __init__(self, model_name="yolov8n.pt", conf_threshold=0.25, device=None):
        """
        Initialize YOLOv8 detector

        Args:
            model_name: YOLOv8 model name (n, s, m, l, x)
            conf_threshold: Confidence threshold for detections
            device: Torch device (will use auto-detection if None)
        """
        self.conf_threshold = conf_threshold
        self.device = device

        # Cache for detection results
        self.detection_cache = {}
        self.cache_size = 5

        # Set device for YOLO
        if device is None:
            # Check if MPS is available (for macOS)
            if torch.backends.mps.is_available():
                self.device = "mps"
            # Otherwise use CUDA or CPU
            else:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load YOLO model
        print(f"Loading YOLOv8 model: {model_name} on device: {self.device}")
        self.model = YOLO(model_name)
        # Set the device for the model
        self.model.to(self.device)

        # Get class names
        self.class_names = self.model.names

    def detect(self, img):
        """
        Detect objects in an image

        Args:
            img: Input BGR image

        Returns:
            results: YOLOv8 results object
            annotated_img: Image with bounding boxes
        """
        # Generate a hash for caching
        img_hash = hash(img.tobytes())

        # Check cache first
        if img_hash in self.detection_cache:
            return self.detection_cache[img_hash]

        # Run inference with specific device and half precision if on GPU
        results = self.model(img,
                            conf=self.conf_threshold,
                            device=self.device,
                            half=self.device != 'cpu',  # Use half precision (FP16) for GPU/MPS
                            verbose=False,              # Reduce output for speed
                            imgsz=320)                  # Smaller inference size for speed

        # Create annotated image
        annotated_img = results[0].plot()

        # Cache the results
        result = (results[0], annotated_img)
        self.detection_cache[img_hash] = result

        # Maintain cache size
        if len(self.detection_cache) > self.cache_size:
            # Remove oldest item
            self.detection_cache.pop(next(iter(self.detection_cache)))

        return result

    def get_relevant_obstacles(self, results, relevant_classes=None):
        """
        Extract relevant obstacle detections from results

        Args:
            results: YOLOv8 results
            relevant_classes: List of class indices to consider as obstacles
                             (if None, consider person, car, bicycle, motorcycle,
                              bus, truck, etc.)

        Returns:
            obstacles: List of obstacle detections (xyxy, conf, cls)
        """
        # Default relevant classes (common obstacles)
        if relevant_classes is None:
            # Common obstacle classes in COCO dataset
            relevant_names = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
                             'fire hydrant', 'stop sign', 'bench', 'chair', 'couch',
                             'potted plant', 'bed', 'dining table', 'toilet', 'tv']

            relevant_classes = [i for i, name in self.class_names.items()
                               if name.lower() in relevant_names]

        # Extract boxes, confidence, and class IDs
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)

            # Filter by relevant classes
            obstacles = []
            for box, conf, cls_id in zip(boxes, confs, class_ids):
                if cls_id in relevant_classes:
                    x1, y1, x2, y2 = box.astype(int)
                    obstacles.append({
                        'box': (x1, y1, x2, y2),
                        'conf': float(conf),
                        'class_id': int(cls_id),
                        'class_name': self.class_names[cls_id]
                    })

            return obstacles

        return []
