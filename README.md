# Obstacle Avoidance System

Real-time monocular depth-based obstacle detection for autonomous navigation using MiDaS and YOLOv8.

## Installation

```bash
git clone https://github.com/shakib75bd/obostacle-avoidance.git
cd obstacle-avoidance
pip install -r requirements.txt
```

## Quick Start

### Real-time Detection

```bash
# Webcam
python main.py --source 0

# Video file
python main.py --source video.mp4 --resolution 320x240
```

### Testing & Metrics

```bash
# Process test video with navigation metrics
python test_video.py --video test_video1.mp4

# Process with custom output and frame skipping for faster analysis
python test_video.py --video test_video1.mp4 --output test_navigation_metrics.mp4 --skip-frames 4

# Real-time webcam testing
python test_video.py --webcam-mode
```

### Report Generation

```bash
# Generate performance reports and navigation accuracy analysis
python evaluation/report_generator.py
```

## Key Features

- **Navigation Decision Accuracy**: Real-time go/stop decisions based on obstacle density
- **Safety Metrics**: False safe/unsafe rate tracking for autonomous driving
- **Multi-source Input**: Webcam and video file support
- **Performance Analysis**: Comprehensive benchmarking vs YOLOv8 baseline

## Project Structure

```
├── main.py              # Real-time obstacle detection
├── test_video.py        # Testing with navigation metrics
├── evaluation/
│   └── report_generator.py  # Performance reports
├── models/              # MiDaS depth + YOLOv8 detection
├── utils/               # Video processing utilities
└── reports/             # Generated analysis
```

## Navigation Metrics

The system tracks navigation-specific performance:

- **Navigation Accuracy**: Correct driving decisions (45-65%)
- **False Safe Rate**: Critical safety metric (<5% target)
- **Detection Rate**: Obstacle detection capability (50-60%)
- **Processing Speed**: Real-time performance (20-30 FPS)

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- Ultralytics YOLOv8
