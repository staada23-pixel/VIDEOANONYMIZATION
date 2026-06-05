"""
Example script demonstrating the Detection Viewer GUI

This script shows how to:
1. Load images and videos
2. Automatically detect faces and hands using MediaPipe
3. Display them in the GUI with adjustable bounding boxes
4. Navigate through video frames with auto-detection for each frame

The GUI follows standard SDK software conventions for facial recognition
and hand detection, automatically adjusting boxes based on detected objects.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.detection import Detection
from gui.detection_viewer import DetectionViewer
from PyQt6.QtWidgets import QApplication


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    
    # Create and show the viewer
    viewer = DetectionViewer()
    viewer.show()
    
    # The viewer is ready to:
    # 1. Load an image - detections will run automatically
    # 2. Load a video - detections will run on each frame
    # 3. Toggle auto-detection with the Auto-Detect button
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
