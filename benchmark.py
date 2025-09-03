import cv2
import numpy as np
import argparse
import time
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Import our modules
from utils.video import VideoSource
from models.depth_estimator import DepthEstimator
from models.object_detector import ObjectDetector
from models.obstacle_map import ObstacleMapGenerator

class BenchmarkEvaluator:
    """
    Evaluates and benchmarks the obstacle detection system
    """
    def __init__(self,
                 depth_model="MiDaS_small",
                 yolo_model="yolov8n.pt",
                 mc_samples=5,
                 output_dir="benchmark_results"):
        """
        Initialize the benchmark evaluator

        Args:
            depth_model: Depth estimation model type
            yolo_model: YOLOv8 model name
            mc_samples: Number of Monte Carlo samples for uncertainty
            output_dir: Directory to save benchmark results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Initialize models
        print("Initializing models...")

        # Depth estimator
        self.depth_model = DepthEstimator(
            model_type=depth_model,
            num_samples=mc_samples
        )

        # Object detector
        self.detector = ObjectDetector(
            model_name=yolo_model,
            conf_threshold=0.25
        )

        # Obstacle map generator
        self.obstacle_generator = ObstacleMapGenerator()

        # Results storage
        self.results = {
            'depth_times': [],
            'detection_times': [],
            'fusion_times': [],
            'total_times': [],
            'frame_count': 0,
            'avg_fps': 0,
            'model_info': {
                'depth_model': depth_model,
                'yolo_model': yolo_model,
                'mc_samples': mc_samples
            }
        }

    def process_frame(self, frame):
        """
        Process a single frame and record timing metrics

        Args:
            frame: Input BGR image

        Returns:
            output: Dictionary with processing results and timing
        """
        frame_start = time.time()

        # 1. Depth estimation with uncertainty
        depth_start = time.time()
        depth_map, uncertainty_map, depth_colored, uncertainty_colored = (
            self.depth_model.estimate_depth_with_uncertainty(frame)
        )
        depth_time = time.time() - depth_start

        # 2. Object detection
        detection_start = time.time()
        results, detection_img = self.detector.detect(frame)
        obstacles = self.detector.get_relevant_obstacles(results)
        detection_time = time.time() - detection_start

        # 3. Generate obstacle map
        fusion_start = time.time()
        obstacle_map, obstacle_viz = self.obstacle_generator.generate_obstacle_map(
            depth_map, uncertainty_map, obstacles, frame.shape
        )
        fusion_time = time.time() - fusion_start

        # Calculate total processing time
        total_time = time.time() - frame_start

        # Record timing metrics
        self.results['depth_times'].append(depth_time)
        self.results['detection_times'].append(detection_time)
        self.results['fusion_times'].append(fusion_time)
        self.results['total_times'].append(total_time)
        self.results['frame_count'] += 1

        return {
            'depth_map': depth_map,
            'uncertainty_map': uncertainty_map,
            'depth_colored': depth_colored,
            'uncertainty_colored': uncertainty_colored,
            'detection_img': detection_img,
            'obstacles': obstacles,
            'obstacle_map': obstacle_map,
            'obstacle_viz': obstacle_viz,
            'timing': {
                'depth_time': depth_time,
                'detection_time': detection_time,
                'fusion_time': fusion_time,
                'total_time': total_time
            }
        }

    def run_benchmark(self, video_source, max_frames=100):
        """
        Run benchmark on a video source

        Args:
            video_source: Path to video file or camera index
            max_frames: Maximum number of frames to process

        Returns:
            benchmark_results: Dictionary with benchmark results
        """
        print(f"Running benchmark on {video_source} for {max_frames} frames...")

        source = 0 if video_source == "0" else video_source
        frame_count = 0

        with VideoSource(source) as video:
            start_time = time.time()

            while frame_count < max_frames:
                ret, frame = video.read()
                if not ret:
                    break

                # Process frame
                self.process_frame(frame)
                frame_count += 1

                # Print progress
                if frame_count % 10 == 0:
                    print(f"Processed {frame_count}/{max_frames} frames")

            total_time = time.time() - start_time

        # Calculate summary statistics
        self.results['avg_fps'] = frame_count / total_time

        # Save results
        self.save_results()

        return self.results

    def save_results(self):
        """Save benchmark results to JSON and generate plots"""
        # Calculate summary statistics
        summary = {
            'avg_depth_time': np.mean(self.results['depth_times']) * 1000,  # ms
            'avg_detection_time': np.mean(self.results['detection_times']) * 1000,  # ms
            'avg_fusion_time': np.mean(self.results['fusion_times']) * 1000,  # ms
            'avg_total_time': np.mean(self.results['total_times']) * 1000,  # ms
            'avg_fps': self.results['avg_fps'],
            'frame_count': self.results['frame_count'],
            'model_info': self.results['model_info']
        }

        # Save summary to JSON
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        result_file = self.output_dir / f"benchmark_results_{timestamp}.json"

        with open(result_file, 'w') as f:
            json.dump(summary, f, indent=4)

        print(f"Saved benchmark results to {result_file}")

        # Generate timing plots
        self._generate_plots(timestamp)

        return summary

    def _generate_plots(self, timestamp):
        """Generate plots from benchmark results"""
        # Timing distribution plot
        plt.figure(figsize=(12, 6))

        # Convert times to ms
        depth_times = np.array(self.results['depth_times']) * 1000
        detection_times = np.array(self.results['detection_times']) * 1000
        fusion_times = np.array(self.results['fusion_times']) * 1000
        total_times = np.array(self.results['total_times']) * 1000

        # Box plot
        plt.subplot(1, 2, 1)
        plt.boxplot(
            [depth_times, detection_times, fusion_times, total_times],
            labels=['Depth', 'Detection', 'Fusion', 'Total']
        )
        plt.ylabel('Time (ms)')
        plt.title('Processing Time Distribution')

        # Bar plot of averages
        plt.subplot(1, 2, 2)
        labels = ['Depth', 'Detection', 'Fusion', 'Total']
        avgs = [
            np.mean(depth_times),
            np.mean(detection_times),
            np.mean(fusion_times),
            np.mean(total_times)
        ]
        plt.bar(labels, avgs)
        plt.ylabel('Average Time (ms)')
        plt.title(f'Average Processing Times (FPS: {self.results["avg_fps"]:.1f})')

        # Save plot
        plot_file = self.output_dir / f"benchmark_plot_{timestamp}.png"
        plt.tight_layout()
        plt.savefig(plot_file)
        print(f"Saved benchmark plot to {plot_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Depth-Based Obstacle Detection")
    parser.add_argument("--source", type=str, default="0",
                        help="Video source (0 for webcam, or path to video file)")
    parser.add_argument("--depth-model", type=str, default="MiDaS_small",
                        help="Depth model type")
    parser.add_argument("--yolo-model", type=str, default="yolov8n.pt",
                        help="YOLOv8 model size (n, s, m, l, x)")
    parser.add_argument("--mc-samples", type=int, default=5,
                        help="Number of Monte Carlo dropout samples")
    parser.add_argument("--max-frames", type=int, default=100,
                        help="Maximum number of frames to process")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Directory to save benchmark results")

    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()

    # Check if video source exists if it's a file
    if args.source != "0" and not os.path.exists(args.source):
        print(f"Error: Video file '{args.source}' does not exist.")
        print("Please provide a valid video file path or use '0' for webcam.")
        return

    # Initialize benchmark evaluator
    evaluator = BenchmarkEvaluator(
        depth_model=args.depth_model,
        yolo_model=args.yolo_model,
        mc_samples=args.mc_samples,
        output_dir=args.output_dir
    )

    # Run benchmark
    results = evaluator.run_benchmark(
        video_source=args.source,
        max_frames=args.max_frames
    )

    # Print summary
    print("\nBenchmark Summary:")
    print(f"Average FPS: {results['avg_fps']:.2f}")
    print(f"Depth estimation time: {np.mean(results['depth_times'])*1000:.1f} ms")
    print(f"Object detection time: {np.mean(results['detection_times'])*1000:.1f} ms")
    print(f"Fusion time: {np.mean(results['fusion_times'])*1000:.1f} ms")
    print(f"Total processing time: {np.mean(results['total_times'])*1000:.1f} ms")
    print(f"Total frames processed: {results['frame_count']}")

if __name__ == "__main__":
    main()
