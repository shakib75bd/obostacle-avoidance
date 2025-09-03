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

        # Load YOLO model
        print(f"Loading YOLOv8 model: {model_name}")
        self.model = YOLO(model_name)

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
        # Run inference
        results = self.model(img, conf=self.conf_threshold)

        # Create annotated image
        annotated_img = results[0].plot()

        return results[0], annotated_img

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
