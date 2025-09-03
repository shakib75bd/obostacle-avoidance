import cv2
import numpy as np
import argparse
import time
import os
from pathlib import Path
import urllib.request

# Import our modules
from utils.video import VideoSource, FPSCounter
from models.depth_estimator import DepthEstimator
from models.object_detector import ObjectDetector
from models.obstacle_map import ObstacleMapGenerator
from utils.visualization import create_visualization, PerformanceMetrics

def parse_args():
    parser = argparse.ArgumentParser(description="Test Obstacle Detection on Sample Videos")
    parser.add_argument("--video", type=str, default="",
                        help="Path to test video file (if empty, will download a sample)")
    parser.add_argument("--download-sample", action="store_true",
                        help="Download a sample test video")
    parser.add_argument("--resolution", type=str, default="320x240",
                        help="Processing resolution (WxH)")
    parser.add_argument("--depth-model", type=str, default="MiDaS_small",
                        help="Depth model type")
    parser.add_argument("--yolo-model", type=str, default="yolov8n.pt",
                        help="YOLOv8 model")
    parser.add_argument("--mc-samples", type=int, default=2,
                        help="Number of Monte Carlo dropout samples")
    parser.add_argument("--output", type=str, default="output.mp4",
                        help="Output video path")
    parser.add_argument("--webcam-mode", action="store_true",
                        help="Optimize for real-time webcam processing")
    parser.add_argument("--save-frames", action="store_true",
                        help="Save individual frames to output_frames directory")
    parser.add_argument("--skip-frames", type=int, default=3,
                        help="Skip N frames for each processed frame to increase speed")
    parser.add_argument("--target-fps", type=int, default=24,
                        help="Target FPS for live processing")
    parser.add_argument("--webcam-source", type=int, default=0,
                        help="Webcam source index (0 for default camera, 1 for external, etc.)")
    parser.add_argument("--list-webcams", action="store_true",
                        help="List all available webcam sources and exit")
    parser.add_argument("--navigation", action="store_true", default=True,
                        help="Enable navigation suggestions (forward or turn)")
    parser.add_argument("--navigation-threshold", type=float, default=0.4,
                        help="Obstacle density threshold for navigation decisions (0.0-1.0)")

    return parser.parse_args()

def download_sample_video():
    """Download a sample video for testing"""
    # List of potential sample videos (urban/driving scenarios with obstacles)
    sample_urls = [
        # Driving scene from Pexels (change to a valid URL before using)
        "https://www.pexels.com/download/video/854213/?h=720&w=1280",
    ]

    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Download the first sample
    sample_path = data_dir / "sample_video.mp4"
    if not sample_path.exists():
        print(f"Downloading sample video to {sample_path}...")
        try:
            urllib.request.urlretrieve(sample_urls[0], sample_path)
            print("Download complete!")
        except Exception as e:
            print(f"Error downloading sample video: {e}")
            print("Please provide your own test video using --video argument")
            return None
    else:
        print(f"Sample video already exists at {sample_path}")

    return str(sample_path)

def list_available_webcams():
    """List all available webcam sources"""
    print("\nChecking available webcam sources...")
    available_sources = []

    # Try to open each webcam source from 0 to 10
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Get camera info if possible
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)

            available_sources.append(i)
            print(f"  Webcam source {i}: {width}x{height} @ {fps}fps")

        cap.release()

    if not available_sources:
        print("  No webcam sources found!")

    print(f"Found {len(available_sources)} webcam sources\n")
    return available_sources

def main():
    # Parse arguments
    args = parse_args()

    # If user requested to list webcams, do that and exit
    if args.list_webcams:
        list_available_webcams()
        return

    # Check if we're using webcam
    is_webcam = args.video == "" and not args.download_sample

    # If we're using webcam, validate the source
    if is_webcam:
        # Quick check if the specified webcam source is available
        cap = cv2.VideoCapture(args.webcam_source)
        if not cap.isOpened():
            print(f"Error: Could not open webcam source {args.webcam_source}")
            print("Use --list-webcams to see available webcam sources")
            cap.release()
            return
        cap.release()

    # Download sample video if requested or if no video specified
    video_path = args.video
    if args.download_sample or (not video_path and not is_webcam):
        sample_path = download_sample_video()
        if sample_path:
            video_path = sample_path
        else:
            print("No video provided and failed to download sample. Exiting.")
            return

    # Parse resolution
    try:
        width, height = map(int, args.resolution.split("x"))
        resolution = (width, height)
    except:
        print(f"Invalid resolution format: {args.resolution}, using default 320x240")
        resolution = (320, 240)

    # Override with requested resolution for higher FPS (240x320)
    resolution = (320, 240)

    # Initialize output frames directory if needed
    if args.save_frames:
        frames_dir = Path("output_frames")
        frames_dir.mkdir(exist_ok=True)
        print(f"Will save frames to {frames_dir}")

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
    obstacle_generator = ObstacleMapGenerator()

    # Performance metrics
    metrics = PerformanceMetrics()
    fps_counter = FPSCounter()

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # If we're in webcam mode, use the target fps
    # If using a video file, use the video's original fps
    if is_webcam:
        output_fps = args.target_fps
    else:
        # We'll read the fps from the video source
        with VideoSource(video_path) as temp_video:
            original_fps = temp_video.get_fps()
            output_fps = original_fps if original_fps > 0 else args.target_fps

    print(f"Output video will use {output_fps} fps")

    out = cv2.VideoWriter(
        args.output,
        fourcc,
        output_fps,  # Use appropriate FPS based on source
        (resolution[0] * 2, resolution[1] * 2)  # Width, height of visualization
    )

    print(f"Processing video: {video_path if not is_webcam else f'Webcam (source: {args.webcam_source})'}")
    print(f"Output will be saved to: {args.output}")

    # Set source for VideoSource
    source = args.webcam_source if is_webcam else video_path

    # Process video
    with VideoSource(source, resolution, webcam_mode=args.webcam_mode) as video:
        frame_idx = 0
        processed_frames = 0
        skipped_frames = 0

        # For dynamic frame skipping to maintain target FPS
        target_fps = args.target_fps
        frame_time_budget = 1.0 / target_fps
        dynamic_skip = args.skip_frames
        frame_start_time = time.time()

        while True:
            current_time = time.time()
            elapsed = current_time - frame_start_time

            # Read frame
            ret, frame = video.read()
            if not ret:
                break

            # Dynamic frame skipping to maintain target FPS
            should_process = True
            if processed_frames > 0:
                # Determine skip behavior based on source type
                if is_webcam:
                    if args.webcam_mode:
                        # In webcam mode with real-time optimization,
                        # we don't need dynamic skipping as VideoSource handles it
                        skip_target = 0
                    else:
                        # For webcam without real-time mode, use fixed skip rate
                        skip_target = args.skip_frames
                else:
                    # For video files, use dynamic skipping to maintain target fps
                    if elapsed < frame_time_budget:
                        # We're ahead of schedule, use requested skip rate
                        skip_target = args.skip_frames
                    else:
                        # We're behind schedule, calculate required skip
                        # Skip more frames to catch up
                        skip_target = max(args.skip_frames, int(elapsed / frame_time_budget))
                        if skip_target > 10:  # Cap maximum skip to prevent huge jumps
                            skip_target = 10
                    # If processing is taking too long, increase skipping
                    if elapsed < frame_time_budget:
                        # We're ahead of schedule, use requested skip rate
                        skip_target = args.skip_frames
                    else:
                        # We're behind schedule, calculate required skip
                        # Skip more frames to catch up
                        skip_target = max(args.skip_frames, int(elapsed / frame_time_budget))
                        if skip_target > 10:  # Cap maximum skip to prevent huge jumps
                            skip_target = 10

                    # Apply skipping logic
                    if skipped_frames < skip_target:
                        skipped_frames += 1
                        should_process = False
                    else:
                        skipped_frames = 0
                        frame_start_time = current_time  # Reset timer

            # For first frame, initialize the timer
            if processed_frames == 0:
                frame_start_time = current_time

            # Skip processing if needed
            if not should_process:
                continue

            processed_frames += 1

            # Update progress
            frame_idx += 1
            if frame_idx % 10 == 0:
                total_frames = video.get_frame_count()
                if total_frames > 0:
                    print(f"Processing frame {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%)")
                else:
                    print(f"Processing frame {frame_idx}")

            # Measure processing time
            frame_start = time.time()

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

            # 4. Determine navigation suggestion based on obstacle density if enabled
            nav_direction = None
            nav_confidence = None
            if args.navigation:
                nav_direction, nav_confidence = obstacle_generator.determine_navigation_direction(
                    obstacle_map, threshold=args.navigation_threshold
                )

            fusion_time = time.time() - fusion_start
            metrics.update('fusion_time', fusion_time)            # Calculate total processing time
            total_time = time.time() - frame_start
            metrics.update('total_time', total_time)

            # Update FPS counter
            fps_counter.update()
            fps = fps_counter.get_fps()
            metrics.update('fps', fps)

            # Don't display effective FPS anymore as it's confusing
            # Just track frame skipping rate
            skip_rate = skipped_frames if skipped_frames > 0 else dynamic_skip
            metrics.update('skip_rate', skip_rate)

            # Create visualization
            visualization = create_visualization(
                frame, depth_colored, uncertainty_colored,
                detection_img, obstacle_viz, fps, metrics,
                webcam_mode=args.webcam_mode,
                navigation_direction=nav_direction,
                navigation_confidence=nav_confidence
            )

            # Write to output video
            out.write(visualization)

            # Save individual frames if requested
            if args.save_frames and frame_idx % 5 == 0:  # Save every 5th frame
                frame_path = f"output_frames/frame_{frame_idx:04d}.jpg"
                cv2.imwrite(frame_path, visualization)

            # Display result (comment out for faster processing)
            # Skip frames for display to improve performance
            if frame_idx % 2 == 0:  # Only show every other frame
                cv2.imshow("Processing Video", visualization)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    # Release resources
    cv2.destroyAllWindows()
    out.release()

    # Print performance summary
    print("\nProcessing complete!")
    print(f"Output saved to: {args.output}")
    print("\nPerformance Summary:")
    summary = metrics.get_summary()
    for metric, value in summary.items():
        if 'time' in metric:
            print(f"{metric}: {value*1000:.1f} ms")
        else:
            print(f"{metric}: {value:.2f}")

if __name__ == "__main__":
    main()
