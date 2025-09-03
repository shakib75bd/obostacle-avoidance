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

### Test Video Processing System

The `test_video.py` script allows for testing and visualization of the obstacle detection system with various options for different use cases.

#### Basic Usage

Process a video file:

```
python test_video.py --video path/to/video.mp4
```

Use webcam (default camera):

```
python test_video.py --webcam-mode
```

List all available webcams:

```
python test_video.py --list-webcams
```

Use a specific webcam (e.g., external camera):

```
python test_video.py --webcam-mode --webcam-source 1
```

#### Processing Modes

1. **Video File Mode**

   Process a video file with standard settings:

   ```
   python test_video.py --video test_video1.mp4
   ```

   Optimize processing with lower resolution and fewer MC samples:

   ```
   python test_video.py --video test_video1.mp4 --resolution 320x240 --mc-samples 2 --target-fps 24
   ```

2. **Webcam Mode**

   Real-time processing with default webcam:

   ```
   python test_video.py --webcam-mode
   ```

   Use external webcam with optimization for real-time performance:

   ```
   python test_video.py --webcam-mode --webcam-source 1 --mc-samples 1 --skip-frames 2
   ```

#### Command-line Arguments

- `--video`: Path to video file for processing
- `--download-sample`: Download a sample test video if none provided
- `--resolution`: Processing resolution in WxH format (default: "320x240")
- `--depth-model`: Depth model type (default: "MiDaS_small")
- `--yolo-model`: YOLOv8 model path (default: "yolov8n.pt")
- `--mc-samples`: Number of Monte Carlo dropout samples (default: 2)
- `--output`: Output video path (default: "output.mp4")
- `--webcam-mode`: Optimize for real-time webcam processing
- `--save-frames`: Save individual frames to output_frames directory
- `--skip-frames`: Skip N frames for each processed frame to increase speed (default: 3)
- `--target-fps`: Target FPS for live processing (default: 24)
- `--webcam-source`: Webcam source index (0 for default camera, 1+ for external cameras)
- `--list-webcams`: List all available webcam sources and exit
- `--navigation`: Enable navigation suggestions based on obstacle density (default: enabled)
- `--navigation-threshold`: Obstacle density threshold for navigation decisions (default: 0.4)

### Navigation Feature

The system includes an obstacle density analyzer that provides navigation guidance based on the density of obstacles in front of the camera:

```
python test_video.py --webcam-mode --navigation
```

The navigation feature:

- Displays a real-time obstacle density gauge showing how blocked the path ahead is
- Recommends "MOVE FORWARD" when the path is clear (obstacle density below threshold)
- Recommends "TURN - PATH BLOCKED" when obstacles are detected (density above threshold)
- Shows path status as "SAFE" or "UNSAFE" based on obstacle density

You can adjust the sensitivity of obstacle detection with:

```
python test_video.py --webcam-mode --navigation-threshold 0.3
```

Lower threshold values (0.2-0.3) make the system more cautious (suggesting turns earlier), while higher values (0.5-0.7) allow navigation through more complex environments.

### Running Benchmarks

To benchmark the system performance:

```
python benchmark.py --source test_video1.mp4 --max-frames 100
```

Replace `test_video1.mp4` with the path to your test video file, or use `0` for webcam:

```
python benchmark.py --source 0 --max-frames 50
```

This will:

1. Process the specified number of frames
2. Record timing metrics for each component
3. Generate summary statistics and plots
4. Save results to the `benchmark_results` directory

## Webcam Mode Features

The webcam mode is designed for real-time obstacle detection and provides several optimizations:

### Webcam Source Selection

The system can automatically detect and list available webcam sources:

```
python test_video.py --list-webcams
```

This will display all connected cameras with their resolution and frame rate information, allowing you to select the most appropriate camera for your use case.

### Real-time Optimization

When using `--webcam-mode`, the system:

1. Uses an optimized frame buffer to minimize latency
2. Adapts frame skipping dynamically based on processing capabilities
3. Prioritizes the most recent frames for processing
4. Automatically detects webcam properties (resolution, FPS)
5. Provides visual feedback with a webcam mode indicator

### Performance Tuning

Fine-tune webcam performance with these options:

- Reduce `--mc-samples` (1-2 recommended for webcam) to speed up depth estimation
- Adjust `--skip-frames` to balance smoothness and processing load
- Set `--target-fps` to match your display capabilities
- Use `--resolution` to reduce the processing resolution for faster performance

### Example Configurations

For maximum responsiveness on slower hardware:

```
python test_video.py --webcam-mode --mc-samples 1 --resolution 160x120 --skip-frames 4
```

For balanced performance on standard hardware:

```
python test_video.py --webcam-mode --mc-samples 2 --resolution 320x240 --skip-frames 2
```

For higher quality on powerful hardware:

```
python test_video.py --webcam-mode --mc-samples 3 --resolution 640x480 --skip-frames 1
```

## System Architecture

### Video Input

- The system accepts input from a webcam or video file
- Low resolution (default: 320x240) for faster processing
- Supports multiple webcam sources with the `--webcam-source` parameter
- Thread-safe frame buffering for real-time applications

### Video Processing Components

#### VideoSource Class

The `VideoSource` class in `utils/video.py` provides a unified interface for handling both video files and webcam inputs:

- Thread-safe frame capturing with queue-based buffering
- Automatic FPS detection for videos and webcams
- Real-time mode with frame skipping for low-latency processing
- Proper resource management with context manager support
- Automatic adaption to source properties

#### FPSCounter

The `FPSCounter` class provides real-time performance measurement:

- Rolling average FPS calculation
- Performance metrics tracking
- Adaptive frame skipping based on processing capabilities

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
