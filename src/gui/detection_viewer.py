"""
================================================================================
DETECTION VIEWER GUI - EDUCATIONAL GUIDE FOR BEGINNERS
================================================================================

WHAT IS THIS MODULE?
This is the graphical user interface (GUI) for viewing images and videos
with automatic face and hand detection.

WHAT IS A GUI?
GUI = Graphical User Interface
- A way for users to interact with your program using buttons, images, etc.
- Instead of typing commands, users click buttons
- Much more user-friendly!

WHAT DOES THIS GUI DO?
1. Displays images and video frames
2. Shows detected faces and hands with bounding boxes
3. Lets users load images/videos by clicking buttons
4. Shows details about detections in a table

LIBRARIES USED:
- PyQt6: For creating the GUI (buttons, windows, etc.)
- OpenCV (cv2): For image processing and loading files
- NumPy: For working with image data as arrays
- MediaPipe: For face and hand detection

THE GUI LAYOUT:
┌─────────────────────────────────────────────────────────────────┐
│ Detection Viewer                                                │
├─────────────────────────────────┬───────────────────────────────┤
│                                 │ Control Buttons               │
│                                 │ - Load Image                  │
│  IMAGE DISPLAY                  │ - Load Video                  │
│  (Shows image with              │ - Auto-Detect ON/OFF          │
│   bounding boxes)               │                               │
│                                 │ Detection Table               │
│                                 │ (Lists all detections)        │
│                                 │                               │
│  [Slider for video frames]      │ Statistics                    │
│                                 │ (How many, confidence, etc.)  │
└─────────────────────────────────┴───────────────────────────────┘

================================================================================
"""

import sys
import logging
from pathlib import Path
from typing import List, Optional

# Configure logging to see what's happening
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - [GUI] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QFileDialog, QStatusBar, QTableWidget,
    QTableWidgetItem, QMessageBox
)
from PyQt6.QtGui import QImage, QPixmap, QColor, QPainter, QPen, QFont
from PyQt6.QtCore import Qt, QTimer

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.detection import Detection

logger.info("Loading detector...")
try:
    from detector.face_hand_detector import FaceHandDetector
    DETECTOR_AVAILABLE = True
    logger.info("✓ Detector module loaded successfully")
except ImportError as e:
    logger.error(f"✗ Failed to load detector: {e}")
    DETECTOR_AVAILABLE = False
    FaceHandDetector = None


class DetectionViewer(QMainWindow):
    """
    MAIN GUI WINDOW CLASS - DETAILED EXPLANATION
    
    INHERITANCE:
    This class "inherits" from QMainWindow, which means it has all the
    properties and methods of QMainWindow (a main application window).
    Think of it like inheriting traits from your parents.
    
    WHAT THIS CLASS DOES:
    1. Creates the main window of the application
    2. Handles loading images and videos
    3. Displays images with detection bounding boxes
    4. Manages the UI elements (buttons, tables, sliders)
    5. Communicates with the face/hand detector
    
    ATTRIBUTES (Instance Variables):
    - current_image: The image currently being displayed
    - current_detections: List of faces/hands found in current image
    - current_frame_idx: Which frame we're viewing (for videos)
    - video_capture: OpenCV object for reading video files
    - detector: The FaceHandDetector instance for AI detection
    - auto_detect_enabled: Whether automatic detection is on/off
    """
    
    def __init__(self):
        """
        CONSTRUCTOR - Initialize the GUI window
        
        This method is called when you create a new DetectionViewer:
            viewer = DetectionViewer()
        
        What happens inside:
        1. Initialize the parent QMainWindow class
        2. Set window title and size
        3. Create instance variables to store data
        4. Load the face/hand detector
        5. Build the user interface
        """
        
        logger.info("=" * 80)
        logger.info("CREATING DETECTION VIEWER GUI")
        logger.info("=" * 80)
        
        # Call parent class constructor
        super().__init__()
        
        # Set window title (what appears in the title bar)
        self.setWindowTitle("Detection Viewer - AI Face & Hand Detection")
        
        # Set initial window size (x, y, width, height)
        # (100, 100) = position on screen
        # 1200, 800 = window size in pixels
        self.setGeometry(100, 100, 1200, 800)
        
        logger.info("Window initialized: 1200x800")
        
        # INSTANCE VARIABLES
        # These store the current state of the application
        
        # Current image being displayed (as numpy array)
        self.current_image = None
        logger.debug("current_image initialized to None")
        
        # List of detected faces/hands in the current image
        # Each item is a Detection object
        self.current_detections: List[Detection] = []
        logger.debug("current_detections initialized as empty list")
        
        # Current frame number (for videos)
        # Frame 0 = first frame, Frame 1 = second frame, etc.
        self.current_frame_idx = 0
        
        # OpenCV VideoCapture object (for reading video files)
        # This is None until we load a video
        self.video_capture = None
        logger.debug("video_capture initialized to None")
        
        # List of frames from video (we cache them here)
        self.video_frames = []
        
        # =====================================================================
        # INITIALIZE THE DETECTOR
        # =====================================================================
        # The detector does the actual face/hand detection
        logger.info("Initializing face/hand detector...")
        
        self.detector = None  # Start with None
        self.auto_detect_enabled = False  # Disabled until detector loads
        
        if DETECTOR_AVAILABLE:
            try:
                # Create a new FaceHandDetector instance
                # This loads the AI models (might take a second)
                self.detector = FaceHandDetector(min_detection_confidence=0.5)
                self.auto_detect_enabled = True
                logger.info("✓ Detector loaded successfully - auto-detection enabled")
                
            except ImportError as e:
                logger.error(f"✗ Failed to initialize detector: {e}")
                logger.error("  Face and hand detection will not work")
                logger.error("  To fix: pip install mediapipe")
        else:
            logger.warning("Detector module not available - detection disabled")
        
        # Now build the GUI
        logger.info("Building GUI...")
        self.init_ui()
        logger.info("✓ GUI fully initialized")
        logger.info("=" * 80)
        
    def init_ui(self):
        """Initialize the user interface"""
        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        
        # Left side - Image display
        left_layout = QVBoxLayout()
        self.image_label = QLabel()
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setScaledContents(False)
        self.image_label.setStyleSheet("border: 1px solid black; background-color: #222;")
        left_layout.addWidget(self.image_label)
        
        # Slider for video/image navigation
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMaximum(0)
        self.slider.valueChanged.connect(self.on_slider_changed)
        left_layout.addWidget(self.slider)
        
        # Right side - Controls and detection list
        right_layout = QVBoxLayout()
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        load_image_btn = QPushButton("Load Image")
        load_image_btn.clicked.connect(self.load_image)
        button_layout.addWidget(load_image_btn)
        
        load_video_btn = QPushButton("Load Video")
        load_video_btn.clicked.connect(self.load_video)
        button_layout.addWidget(load_video_btn)
        
        self.auto_detect_btn = QPushButton("Auto-Detect: ON" if self.auto_detect_enabled else "Auto-Detect: OFF (Install MediaPipe)")
        self.auto_detect_btn.setStyleSheet("background-color: #90EE90;" if self.auto_detect_enabled else "background-color: #FFB6C6;")
        self.auto_detect_btn.clicked.connect(self.toggle_auto_detect)
        self.auto_detect_btn.setEnabled(self.auto_detect_enabled)
        button_layout.addWidget(self.auto_detect_btn)
        
        right_layout.addLayout(button_layout)
        
        # Detection list table
        label = QLabel("Detections in current frame:")
        label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        right_layout.addWidget(label)
        
        self.detection_table = QTableWidget()
        self.detection_table.setColumnCount(4)
        self.detection_table.setHorizontalHeaderLabels(["Label", "Confidence", "Position", "Size"])
        self.detection_table.setMaximumHeight(300)
        right_layout.addWidget(self.detection_table)
        
        # Statistics
        self.stats_label = QLabel("No data loaded")
        self.stats_label.setStyleSheet("padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        right_layout.addWidget(self.stats_label)
        
        right_layout.addStretch()
        
        # Add layouts to main layout
        main_layout.addLayout(left_layout, 3)
        main_layout.addLayout(right_layout, 1)
        
        main_widget.setLayout(main_layout)
        
        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")
        
    def load_image(self):
        """
        LOAD AN IMAGE FILE - STEP BY STEP
        
        This method is called when user clicks "Load Image" button.
        
        WHAT HAPPENS:
        1. Open file dialog (let user choose a file)
        2. Load the image using OpenCV
        3. Run face/hand detection if enabled
        4. Display the image with detections
        5. Update the status bar
        
        WHY cv2.imread()?
        - cv2 = OpenCV library
        - imread() = "image read"
        - It loads an image file from disk into a numpy array
        """
        
        logger.info("-" * 80)
        logger.info("LOAD IMAGE - User clicked 'Load Image' button")
        logger.info("-" * 80)
        
        # Open file dialog for user to select an image
        # file_path = path to the image they selected (or None if cancelled)
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Image",  # Window title
            "",  # Default directory (empty = current directory)
            "Image Files (*.png *.jpg *.jpeg *.bmp)"  # File type filter
        )
        
        # Check if user actually selected a file (not clicked Cancel)
        if file_path:
            logger.info(f"User selected: {file_path}")
            
            # LOAD IMAGE
            # cv2.imread() reads the image file and returns a numpy array
            # The array has shape (height, width, 3) where 3 = BGR channels
            self.current_image = cv2.imread(file_path)
            logger.debug(f"Image loaded: shape={self.current_image.shape}")
            
            # For single images, we don't have a video, so:
            self.video_frames = [self.current_image]  # Put image in list
            self.current_frame_idx = 0  # We're viewing frame 0 (the only frame)
            self.slider.setMaximum(0)  # Slider has only one position
            
            # ================================================================
            # AUTO-DETECT FACES AND HANDS
            # ================================================================
            if self.auto_detect_enabled and self.detector:
                logger.info("Running face/hand detection...")
                try:
                    # Call the detector's detect() method
                    # This returns a list of Detection objects
                    self.current_detections = self.detector.detect(self.current_image)
                    logger.info(f"✓ Detection complete: {len(self.current_detections)} objects found")
                except Exception as e:
                    logger.error(f"✗ Detection failed: {e}")
                    logger.error("Continuing without detections...")
                    self.current_detections = []
            else:
                logger.info("Auto-detection disabled or detector not available")
                self.current_detections = []
            
            # Display the image on screen
            self.display_frame()
            
            # Update status bar with information
            status = f"Loaded: {Path(file_path).name} | Detected: {len(self.current_detections)} objects"
            self.statusBar.showMessage(status)
            logger.info(status)
            
        else:
            logger.info("User cancelled image selection")

            
    def load_video(self):
        """Load a video file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        
        if file_path:
            self.video_capture = cv2.VideoCapture(file_path)
            frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            
            self.slider.setMaximum(frame_count - 1)
            self.slider.setValue(0)
            
            self.video_frames = []
            self.current_frame_idx = 0
            
            self.load_frame(0)
            self.statusBar.showMessage(f"Loaded: {Path(file_path).name} ({frame_count} frames)")
            
    def load_frame(self, frame_idx: int):
        """Load a specific frame from video"""
        if self.video_capture:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.video_capture.read()
            if ret:
                self.current_image = frame
                self.current_frame_idx = frame_idx
                
                # Auto-detect faces and hands if enabled and available
                if self.auto_detect_enabled and self.detector:
                    self.current_detections = self.detector.detect(self.current_image)
                
                self.display_frame()
                
    def on_slider_changed(self, value: int):
        """Handle slider change"""
        if self.video_capture:
            self.load_frame(value)
            
    def toggle_auto_detect(self):
        """Toggle automatic face/hand detection"""
        if not self.detector:
            QMessageBox.warning(
                self, 
                "MediaPipe Not Available", 
                "Please install MediaPipe:\npython -m pip install mediapipe"
            )
            return
        
        self.auto_detect_enabled = not self.auto_detect_enabled
        
        if self.auto_detect_enabled:
            self.auto_detect_btn.setText("Auto-Detect: ON")
            self.auto_detect_btn.setStyleSheet("background-color: #90EE90;")
            
            # Re-detect current image if one is loaded
            if self.current_image is not None:
                self.current_detections = self.detector.detect(self.current_image)
                self.display_frame()
        else:
            self.auto_detect_btn.setText("Auto-Detect: OFF")
            self.auto_detect_btn.setStyleSheet("background-color: #FFB6C6;")
            self.current_detections = []
            self.display_frame()
            
    def display_frame(self):
        """
        DISPLAY THE CURRENT IMAGE WITH DETECTION BOUNDING BOXES
        
        This is where the magic happens - we draw boxes around detected faces/hands!
        
        WHAT HAPPENS (PIPELINE):
        1. Copy the current image (so we don't modify the original)
        2. For each detected object, draw a bounding box and label
        3. Convert image colors from BGR (OpenCV) to RGB (Qt)
        4. Resize image to fit the display label
        5. Convert numpy array to Qt image format
        6. Display on screen
        7. Update the detection table
        
        WHY COPY THE IMAGE?
        If we draw on the original, we modify it permanently.
        Each time we display it, more boxes get drawn on top of old ones!
        Copying ensures we always start fresh from the original image.
        """
        
        # Check if we have an image to display
        if self.current_image is None:
            logger.warning("No image to display")
            return
        
        logger.debug("Displaying frame with detections...")
        
        # STEP 1: Create a copy of the image to draw on
        # original image: unchanged
        # display_image: will be modified with boxes and text
        display_image = self.current_image.copy()
        logger.debug(f"Created copy of image for drawing: {display_image.shape}")
        
        # STEP 2: Draw all detected objects
        # Loop through each detected face/hand
        for detection in self.current_detections:
            self.draw_detection(display_image, detection)
        
        logger.debug(f"Drew {len(self.current_detections)} bounding boxes")
        
        # STEP 3: Convert image colors
        # OpenCV uses BGR format, but Qt (GUI) expects RGB
        # BGR = Blue, Green, Red
        # RGB = Red, Green, Blue
        # They're reversed! So we need to convert
        rgb_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        logger.debug("Converted image from BGR to RGB")
        
        # STEP 4: Get image dimensions
        h, w, ch = rgb_image.shape  # h=height, w=width, ch=channels
        
        # STEP 5: Resize image to fit the display label
        # The label has a fixed size (640x480 approximately)
        # We need to scale the image to fit without being distorted
        max_width = self.image_label.width()   # How many pixels wide is the label?
        max_height = self.image_label.height()  # How many pixels tall is the label?
        
        # Calculate scale factor (how much to shrink/enlarge)
        # We want to fit the entire image in the label
        # scale = min() picks the smaller value so image doesn't go out of bounds
        scale = min(max_width / w, max_height / h)
        
        # Calculate new size
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        logger.debug(f"Resizing image: {w}x{h} → {new_w}x{new_h} (scale={scale:.2f})")
        
        # Resize the image
        rgb_image = cv2.resize(rgb_image, (new_w, new_h))
        
        # STEP 6: Convert numpy array to Qt image format
        # Qt doesn't understand numpy arrays, so we convert
        # Calculate bytes per line (needed for Qt)
        bytes_per_line = 3 * new_w
        
        # Create QImage (Qt's image format)
        q_img = QImage(
            rgb_image.data,  # Pixel data
            new_w,  # Width
            new_h,  # Height
            bytes_per_line,  # Bytes per line
            QImage.Format.Format_RGB888  # Format: RGB, 8 bits per channel
        )
        
        # STEP 7: Display on screen
        # Convert QImage to QPixmap (drawable version)
        pixmap = QPixmap.fromImage(q_img)
        
        # Set the label to show this pixmap
        self.image_label.setPixmap(pixmap)
        logger.debug("✓ Image displayed on screen")
        
        # STEP 8: Update the detection table
        # Show details about each detection in the table on the right
        self.update_detection_table()

        
    def draw_detection(self, image: np.ndarray, detection: Detection):
        """
        DRAW A SINGLE DETECTION (BOUNDING BOX + LABEL) ON THE IMAGE
        
        This method draws one bounding box and label for one detected face/hand.
        
        WHAT HAPPENS:
        1. Extract coordinates from the detection object
        2. Draw the rectangle (bounding box)
        3. Calculate text size
        4. Draw background for the label
        5. Draw the label text
        
        WHY DRAW A BACKGROUND FOR TEXT?
        So the text is readable even if the background is dark/complex.
        The background rectangle provides contrast.
        
        Args:
            image: The image to draw on (modified in place)
            detection: The Detection object with coordinates and label
        """
        
        # Extract coordinates from detection object
        x1, y1 = detection.x, detection.y
        x2, y2 = detection.x + detection.w, detection.y + detection.h
        
        logger.debug(f"Drawing detection: {detection.label} at ({x1},{y1})-({x2},{y2})")
        
        # STEP 1: Draw the bounding box (rectangle)
        color = (0, 255, 0)  # Green color in BGR format (0=B, 255=G, 0=R)
        thickness = 2  # Line thickness in pixels
        
        # cv2.rectangle(image, top_left, bottom_right, color, thickness)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        logger.debug(f"✓ Drew rectangle at ({x1},{y1})-({x2},{y2})")
        
        # STEP 2: Prepare the label text
        # Format: "Label: Confidence"
        # Example: "Face: 0.95"
        label = f"{detection.label}: {detection.confidence:.2f}"
        
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX  # Font style
        font_scale = 0.5  # Font size (0.5 = small)
        thickness = 1  # Text line thickness
        
        # STEP 3: Get text size (needed to draw background)
        # Why? So we know how big the background rectangle should be
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )
        logger.debug(f"Text size: {text_width}x{text_height}")
        
        # STEP 4: Draw background rectangle for the label
        # This creates a colored rectangle behind the text
        # so the text is visible regardless of the image background
        cv2.rectangle(
            image,
            (x1, y1 - text_height - baseline - 4),  # Top-left corner of background
            (x1 + text_width, y1),  # Bottom-right corner
            color,  # Same green color
            -1  # -1 means fill the rectangle (not just outline)
        )
        
        # STEP 5: Draw the label text
        # Put the actual text on top of the background
        cv2.putText(
            image,
            label,  # The text to draw
            (x1, y1 - baseline - 2),  # Position (x, y)
            font,  # Font style
            font_scale,  # Font size
            (0, 0, 0),  # Text color: Black (high contrast with green)
            thickness  # Line thickness
        )
        
        logger.debug(f"✓ Drew label: {label}")

        
    def update_detection_table(self):
        """Update the detection list table"""
        self.detection_table.setRowCount(0)
        
        for detection in self.current_detections:
            row = self.detection_table.rowCount()
            self.detection_table.insertRow(row)
            
            self.detection_table.setItem(row, 0, QTableWidgetItem(detection.label))
            self.detection_table.setItem(row, 1, QTableWidgetItem(f"{detection.confidence:.3f}"))
            self.detection_table.setItem(row, 2, QTableWidgetItem(f"({detection.x}, {detection.y})"))
            self.detection_table.setItem(row, 3, QTableWidgetItem(f"{detection.w}x{detection.h}"))
            
        # Update statistics
        if self.current_detections:
            stats = f"Frame {self.current_frame_idx}: {len(self.current_detections)} detections"
            avg_conf = sum(d.confidence for d in self.current_detections) / len(self.current_detections)
            stats += f"\nAverage confidence: {avg_conf:.3f}"
            self.stats_label.setText(stats)
        else:
            self.stats_label.setText("Frame: No detections")
            
    def set_detections(self, detections: List[Detection]):
        """Set detections to display"""
        self.current_detections = detections
        self.display_frame()
        
    def clear(self):
        """Clear the viewer"""
        self.current_image = None
        self.current_detections = []
        self.image_label.clear()
        self.detection_table.setRowCount(0)
        self.stats_label.setText("No data loaded")


def main():
    app = QApplication(sys.argv)
    viewer = DetectionViewer()
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
