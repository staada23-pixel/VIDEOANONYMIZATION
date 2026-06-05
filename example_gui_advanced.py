"""
Advanced example demonstrating the Detection Viewer GUI with auto-detection

This example shows:
1. Automatic face and hand detection using MediaPipe
2. How the GUI adjusts bounding boxes based on detected faces/hands
3. Real-time detection when loading images
"""

import sys
from pathlib import Path
import numpy as np
import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.detection import Detection
from gui.detection_viewer import DetectionViewer
from PyQt6.QtWidgets import QApplication


def create_sample_image_with_faces():
    """Create a sample image with actual face-like features for testing"""
    # Create a simple image with some shapes
    img = np.ones((480, 640, 3), dtype=np.uint8) * 240
    
    # Draw some face-like shapes (circles representing faces)
    cv2.circle(img, (200, 150), 80, (200, 150, 100), -1)  # Face 1
    cv2.circle(img, (500, 150), 70, (210, 160, 110), -1)  # Face 2
    
    # Add eyes to make them more face-like
    cv2.circle(img, (170, 130), 15, (50, 50, 50), -1)     # Eye 1
    cv2.circle(img, (230, 130), 15, (50, 50, 50), -1)     # Eye 2
    cv2.circle(img, (470, 120), 12, (50, 50, 50), -1)     # Eye 3
    cv2.circle(img, (520, 120), 12, (50, 50, 50), -1)     # Eye 4
    
    # Add some hand-like shapes
    cv2.rectangle(img, (50, 300), (150, 420), (200, 180, 140), -1)  # Hand 1
    cv2.rectangle(img, (500, 320), (600, 430), (210, 190, 150), -1) # Hand 2
    
    # Add some text
    cv2.putText(img, "Sample Image for Face/Hand Detection", (100, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return img


def main():
    """Main entry point with auto-detection"""
    app = QApplication(sys.argv)
    
    # Create viewer
    viewer = DetectionViewer()
    
    # Create and save sample image
    sample_img = create_sample_image_with_faces()
    data_dir = Path(__file__).parent / "data" / "input"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    sample_img_path = data_dir / "sample_for_detection.jpg"
    cv2.imwrite(str(sample_img_path), sample_img)
    
    # Show viewer
    viewer.show()
    
    # Automatically load and detect the sample image
    viewer.current_image = sample_img
    viewer.video_frames = [sample_img]
    
    # Run auto-detection
    if viewer.auto_detect_enabled:
        viewer.current_detections = viewer.detector.detect(sample_img)
    
    viewer.slider.setMaximum(0)
    viewer.display_frame()
    
    num_detections = len(viewer.current_detections)
    viewer.statusBar.showMessage(
        f"Loaded sample image with {num_detections} auto-detected objects (Faces & Hands)"
    )
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

