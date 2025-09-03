import cv2
import numpy as np
import argparse
import time
import os
from pathlib import Path

# Import our modules
from utils.video import VideoSource, FPSCounter
from models.depth_estimator import DepthEstimator
from models.object_detector import ObjectDetector
from models.obstacle_map import ObstacleMapGenerator
from utils.visualization import create_visualization, PerformanceMetrics

def parse_args():
    parser = argparse.ArgumentParser(description="Depth-Based Obstacle Detection with Uncertainty")
    parser.add_argument("--source", type=str, default="0",
                        help="Video source (0 for webcam, or path to video file)")
    parser.add_argument("--resolution", type=str, default="320x240",
                        help="Input resolution (WxH)")
    parser.add_argument("--output", type=str, default="",
                        help="Output video path (empty for no output)")
    parser.add_argument("--depth-model", type=str, default="MiDaS_small",
                        help="Depth model type")
    parser.add_argument("--yolo-model", type=str, default="yolov8n.pt",
                        help="YOLOv8 model size (n, s, m, l, x)")
    parser.add_argument("--mc-samples", type=int, default=5,
                        help="Number of Monte Carlo dropout samples")
    parser.add_argument("--uncertainty-threshold", type=float, default=0.3,
                        help="Uncertainty threshold for confidence regions")
    parser.add_argument("--show-fps", action="store_true",
                        help="Display FPS counter")

    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()

    # Parse resolution
    try:
        width, height = map(int, args.resolution.split("x"))
        resolution = (width, height)
    except:
        print(f"Invalid resolution format: {args.resolution}, using default 320x240")
        resolution = (320, 240)

    # Initialize video source
    source = 0 if args.source == "0" else args.source

    # Initialize models
    print("Initializing models...")

    # Depth estimator with Monte Carlo dropout
    depth_model = DepthEstimator(
        model_type=args.depth_model,
        num_samples=args.mc_samples
    )

    # YOLOv8 object detector
    detector = ObjectDetector(
        model_name=args.yolo_model,
        conf_threshold=0.25
    )

    # Obstacle map generator
    obstacle_generator = ObstacleMapGenerator(
        uncertainty_threshold=args.uncertainty_threshold
    )

    # Performance metrics
    metrics = PerformanceMetrics()
    fps_counter = FPSCounter()

    # Video writer for output
    out = None

    print(f"Opening video source: {source}")
    with VideoSource(source, resolution) as video:
        # Initialize video writer if output path is specified
        if args.output:
            output_path = args.output
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                output_path,
                fourcc,
                30.0,  # FPS
                (resolution[0] * 2, resolution[1] * 2)  # Width, height of visualization
            )
            print(f"Recording output to: {output_path}")

        print("Processing video...")
        while True:
            # Measure total processing time per frame
            frame_start = time.time()

            # Read frame
            ret, frame = video.read()
            if not ret:
                break

            # 1. Depth estimation with uncertainty
            depth_start = time.time()
            depth_map, uncertainty_map, depth_colored, uncertainty_colored = (
                depth_model.estimate_depth_with_uncertainty(frame)
            )
            depth_time = time.time() - depth_start
            metrics.update('depth_time', depth_time)

            # 2. Object detection
            detection_start = time.time()
            results, detection_img = detector.detect(frame)
            obstacles = detector.get_relevant_obstacles(results)
            detection_time = time.time() - detection_start
            metrics.update('detection_time', detection_time)

            # 3. Generate obstacle map
            fusion_start = time.time()
            obstacle_map, obstacle_viz = obstacle_generator.generate_obstacle_map(
                depth_map, uncertainty_map, obstacles, frame.shape
            )
            fusion_time = time.time() - fusion_start
            metrics.update('fusion_time', fusion_time)

            # Calculate total processing time
            total_time = time.time() - frame_start
            metrics.update('total_time', total_time)

            # Update FPS counter
            fps_counter.update()
            fps = fps_counter.get_fps()
            metrics.update('fps', fps)

            # Create visualization
            visualization = create_visualization(
                frame, depth_colored, uncertainty_colored,
                detection_img, obstacle_viz, fps, metrics
            )

            # Display result
            cv2.imshow("Obstacle Detection", visualization)

            # Write to output if specified
            if out is not None:
                out.write(visualization)

            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release resources
    cv2.destroyAllWindows()
    if out is not None:
        out.release()

    # Print performance summary
    print("\nPerformance Summary:")
    summary = metrics.get_summary()
    for metric, value in summary.items():
        if 'time' in metric:
            print(f"{metric}: {value*1000:.1f} ms")
        else:
            print(f"{metric}: {value:.2f}")

if __name__ == "__main__":
    main()
