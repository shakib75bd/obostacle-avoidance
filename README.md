# Depth-Based Obstacle Detection and Avoidance System

This project implements a robust depth-estimation-based obstacle detection system for low-resolution monocular cameras using uncertainty-guided adaptive region fusion.

## Overview

The system combines monocular depth estimation with object detection to create an obstacle likelihood map. The main components are:

1. **Monocular Depth Estimation**: Using MiDaS to estimate depth from a single image
2. **Uncertainty Quantification**: Using Monte Carlo dropout to estimate depth uncertainty
3. **Object Detection**: Using YOLOv8 to detect common obstacles
4. **Adaptive Region Fusion**: Combining depth and detection information based on uncertainty
5. **Obstacle Likelihood Map**: Final output representing obstacle locations and confidence

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- Ultralytics YOLOv8
- Other dependencies in `requirements.txt`

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/obstacle-avoidance.git
   cd obstacle-avoidance
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running the Main System

To run the system on your webcam:

```
python main.py --source 0
```

For a video file:

```
python main.py --source path/to/video.mp4 --resolution 320x240
```

Additional arguments:

- `--depth-model`: Depth model type (default: "MiDaS_small")
- `--yolo-model`: YOLOv8 model (default: "yolov8n.pt")
- `--mc-samples`: Number of Monte Carlo dropout samples (default: 5)
- `--uncertainty-threshold`: Threshold for confidence regions (default: 0.3)
- `--output`: Output video path (default: no output)

### Running Benchmarks

To benchmark the system performance:

```
python benchmark.py --source path/to/video.mp4 --max-frames 100
```

This will:

1. Process the specified number of frames
2. Record timing metrics for each component
3. Generate summary statistics and plots
4. Save results to the `benchmark_results` directory

## System Architecture

### Video Input

- The system accepts input from a webcam or video file
- Low resolution (default: 320x240) for faster processing

### Depth Estimation Module

- Uses MiDaS model for monocular depth estimation
- Implements Monte Carlo dropout for uncertainty estimation
- Outputs both depth and uncertainty maps

### Object Detection Module

- Uses YOLOv8 for detecting common obstacles
- Focuses on relevant classes (person, vehicle, etc.)
- Outputs bounding boxes and confidence scores

### Adaptive Region Fusion

- Separates the image into high and low confidence regions based on depth uncertainty
- Uses depth information in high-confidence regions
- Relies more on object detection in low-confidence regions
- Combines both sources to create a unified obstacle likelihood map

### Visualization

- Shows original image, depth map, detections, and obstacle map
- Displays performance metrics (FPS, processing times)

## Performance Optimization

The system is designed for real-time operation on consumer hardware:

- Uses lightweight models (MiDaS small, YOLOv8n)
- Processes low-resolution images
- Implements efficient fusion algorithms

## Evaluation

The system can be evaluated on:

1. **Processing speed**: FPS, component latencies
2. **Depth accuracy**: Compared to ground truth (if available)
3. **Obstacle detection accuracy**: True/false positives
4. **Overall system robustness**: Performance in challenging conditions

## Customization

- **Depth Model**: Change to other MiDaS variants or custom models
- **Object Detection**: Use different YOLOv8 sizes (n, s, m, l, x) for speed/accuracy tradeoff
- **Uncertainty Threshold**: Adjust to change sensitivity to depth uncertainty
- **Resolution**: Change input resolution based on hardware capabilities

## Limitations

- Depth estimation accuracy limited by monocular vision constraints
- Performance depends on hardware capabilities
- May struggle in extreme lighting conditions or with transparent/reflective objects

## License

This project is licensed under the MIT License - see the LICENSE file for details.
