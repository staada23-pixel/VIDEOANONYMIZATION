"""
================================================================================
SIMPLE GUI TEST - WITHOUT MEDIAPIPE DEPENDENCY
================================================================================

This is a simplified test version that doesn't require MediaPipe.
Use this to test if the GUI itself works correctly.

Once this works, we'll enable MediaPipe detection.

EDUCATIONAL NOTES FOR BEGINNERS:
- This shows how to build a GUI step-by-step
- We start simple, then add features
- This is called "incremental development" - build piece by piece
"""

import sys
import logging
from pathlib import Path
import cv2
import numpy as np

# Set up logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt

sys.path.insert(0, str(Path(__file__).parent / "src"))
from models.detection import Detection


class SimpleDetectionViewer(QMainWindow):
    """
    SIMPLIFIED GUI - For testing without MediaPipe
    
    FEATURES:
    1. Load images
    2. Display images in a window
    3. Show detection boxes (manually created for testing)
    """
    
    def __init__(self):
        logger.info("Creating simple detection viewer...")
        super().__init__()
        
        self.setWindowTitle("Simple Detection Viewer - TEST")
        self.setGeometry(100, 100, 1000, 700)
        
        self.current_image = None
        
        # Create main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Create layout
        layout = QVBoxLayout()
        
        # Image display label
        self.image_label = QLabel()
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet("border: 2px solid black; background-color: gray;")
        layout.addWidget(self.image_label)
        
        # Button layout
        button_layout = QHBoxLayout()
        
        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(self.load_image)
        button_layout.addWidget(load_btn)
        
        test_btn = QPushButton("Load Test Image")
        test_btn.clicked.connect(self.load_test_image)
        button_layout.addWidget(test_btn)
        
        layout.addLayout(button_layout)
        
        main_widget.setLayout(layout)
        
        logger.info("✓ Simple GUI created successfully")
    
    def load_image(self):
        """Load an image from disk"""
        logger.info("User clicked 'Load Image'")
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path:
            logger.info(f"Loading image: {file_path}")
            self.current_image = cv2.imread(file_path)
            
            if self.current_image is None:
                logger.error(f"Failed to load image: {file_path}")
                return
            
            logger.info(f"Image loaded successfully: {self.current_image.shape}")
            self.display_image()
    
    def load_test_image(self):
        """Create and load a test image"""
        logger.info("Creating test image...")
        
        # Create a simple test image
        img = np.ones((480, 640, 3), dtype=np.uint8) * 200
        
        # Add some shapes to make it interesting
        cv2.circle(img, (200, 150), 60, (200, 100, 50), -1)  # Face-like circle
        cv2.circle(img, (500, 150), 50, (200, 100, 50), -1)  # Another circle
        cv2.rectangle(img, (50, 300), (150, 400), (100, 200, 100), -1)  # Green rectangle
        
        # Add some text
        cv2.putText(img, "Test Image", (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        self.current_image = img
        logger.info("Test image created")
        
        # Draw some sample detections
        self.draw_test_boxes()
        
        self.display_image()
    
    def draw_test_boxes(self):
        """Draw test detection boxes on the image"""
        logger.info("Drawing test detection boxes...")
        
        # Create sample detections for testing
        test_detections = [
            Detection(x=120, y=90, w=160, h=120, confidence=0.95, label="Face"),
            Detection(x=420, y=80, w=160, h=140, confidence=0.88, label="Face"),
            Detection(x=20, y=280, w=140, h=120, confidence=0.92, label="Hand"),
        ]
        
        for detection in test_detections:
            x1, y1 = detection.x, detection.y
            x2, y2 = detection.x + detection.w, detection.y + detection.h
            
            # Draw box
            cv2.rectangle(self.current_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{detection.label}: {detection.confidence:.2f}"
            cv2.putText(
                self.current_image, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )
            
            logger.debug(f"Drew box: {detection.label}")
    
    def display_image(self):
        """Display the current image on screen"""
        if self.current_image is None:
            logger.warning("No image to display")
            return
        
        logger.info("Displaying image...")
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        
        # Scale to fit
        max_width = 800
        max_height = 600
        scale = min(max_width / w, max_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        rgb_image = cv2.resize(rgb_image, (new_w, new_h))
        
        # Convert to Qt format
        bytes_per_line = 3 * new_w
        q_img = QImage(rgb_image.data, new_w, new_h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Display
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap)
        
        logger.info("✓ Image displayed on screen")


def main():
    logger.info("=" * 80)
    logger.info("SIMPLE DETECTION VIEWER - TEST VERSION")
    logger.info("=" * 80)
    
    app = QApplication(sys.argv)
    viewer = SimpleDetectionViewer()
    viewer.show()
    
    logger.info("GUI is running. Click 'Load Test Image' to see a demonstration.")
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
