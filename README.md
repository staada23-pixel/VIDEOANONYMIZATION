# Detection Viewer GUI

A modern graphical user interface for viewing and analyzing object detections on images and videos with automatic face and hand detection using MediaPipe.

## Features

- **Automatic Face & Hand Detection**: Uses MediaPipe SDK for automatic detection of faces and hands
- **Load Images & Videos**: Load image files or video files for analysis
- **Display Detections**: Visualize bounding boxes with labels and confidence scores
- **Navigation**: Use sliders to browse through video frames
- **Detection Table**: View detailed information about each detection
- **Statistics**: See detection statistics for each frame
- **Auto-Detect Toggle**: Enable/disable automatic detection as needed

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the GUI

```bash
python example_gui.py
```

Or with automatic face/hand detection:

```bash
python example_gui_advanced.py
```

### Basic Usage

1. **Load an Image or Video**:
   - Click "Load Image" to open an image file
   - Click "Load Video" to open a video file
   - Auto-detection will run automatically if enabled

2. **View Detections**:
   - Detections are displayed as bounding boxes with labels
   - Use the slider to navigate through video frames
   - View detection details in the table on the right
   - Detections update automatically as you browse frames

3. **Toggle Auto-Detection**:
   - Click "Auto-Detect: ON/OFF" to enable or disable automatic detection
   - Green button = Detection enabled
   - Red button = Detection disabled

4. **Detection Information**:
   - **Label**: The class of detected object (Face, Left Hand, Right Hand, etc.)
   - **Confidence**: The detection confidence score (0.0-1.0)
   - **Position**: X, Y coordinates
   - **Size**: Width and height of the bounding box

### Programmatic Usage

```python
from models.detection import Detection
from gui.detection_viewer import DetectionViewer
from detector.face_hand_detector import FaceHandDetector
from PyQt6.QtWidgets import QApplication
import cv2

app = QApplication([])

# Create viewer
viewer = DetectionViewer()

# Load an image
image = cv2.imread("path/to/image.jpg")
viewer.current_image = image

# Auto-detect faces and hands
viewer.current_detections = viewer.detector.detect(image)

# Display
viewer.display_frame()
viewer.show()

app.exec()
```

## GUI Components

### Detection Viewer (Main Window)
- **Image Display**: Shows the current frame with detection bounding boxes
- **Slider**: Navigate through video frames
- **Control Buttons**: 
  - Load Image: Open an image file
  - Load Video: Open a video file
  - Auto-Detect: Toggle automatic face/hand detection
- **Detection Table**: Lists all detections in the current frame
- **Statistics Panel**: Shows frame statistics and average confidence

### Supported Formats

**Images**: PNG, JPG, JPEG, BMP
**Videos**: MP4, AVI, MOV, MKV

## Architecture

The GUI is built with PyQt6, OpenCV, and MediaPipe:

```
src/
├── gui/
│   ├── __init__.py
│   └── detection_viewer.py       # Main GUI implementation
├── detector/
│   └── face_hand_detector.py     # MediaPipe-based face/hand detection
├── models/
│   └── detection.py               # Detection dataclass
└── ...
```

## Detection Data Structure

```python
@dataclass
class Detection:
    x: int              # Left edge of bounding box
    y: int              # Top edge of bounding box
    w: int              # Width of bounding box
    h: int              # Height of bounding box
    confidence: float   # Confidence score (0.0-1.0)
    label: str          # Detection label/class
```

## Face and Hand Detection

The application uses **MediaPipe** for state-of-the-art face and hand detection, following standard SDK software conventions:

### Face Detection
- Detects multiple faces per frame
- Returns bounding boxes with confidence scores
- Works in real-time on images and videos

### Hand Detection
- Detects up to 10 hands per frame
- Identifies left/right hand
- Returns bounding boxes with confidence scores
- Uses hand landmarks for precise localization

### Configuration

```python
detector = FaceHandDetector(min_detection_confidence=0.5)
detections = detector.detect(image)
```

You can adjust the minimum confidence threshold (0.0-1.0) to control detection sensitivity.

## Keyboard Shortcuts

- **Esc**: Exit the application

## Customization

You can customize the GUI appearance by modifying:

- Colors: Change `color = (0, 255, 0)` in `draw_detection()` method
- Font size: Modify `font_scale` parameter
- Box thickness: Change `thickness` parameter
- Detection confidence threshold: Adjust in `FaceHandDetector` initialization

## Performance

- **Images**: Face and hand detection runs instantly
- **Videos**: Detection runs in real-time as frames are loaded
- **GPU Support**: MediaPipe can utilize GPU for faster processing if available

## Troubleshooting

### No detections found
- Ensure the image has clear faces or hands
- Lower the confidence threshold in FaceHandDetector
- Check lighting and image quality

### GUI is slow
- Reduce video resolution
- Lower the confidence threshold
- Disable auto-detection if processing large videos
