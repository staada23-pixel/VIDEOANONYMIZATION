"""
================================================================================
FACE AND HAND DETECTION MODULE - EDUCATIONAL GUIDE FOR BEGINNERS
================================================================================

WHAT IS THIS MODULE?
This module detects faces and hands in images using MediaPipe, an AI/ML 
library from Google. Think of it as giving your program "eyes" to recognize 
where people's faces and hands are in a picture.

HOW DOES IT WORK? (Pipeline Overview)
1. Load an image (JPG, PNG, etc.)
2. Convert the image to RGB format (MediaPipe needs RGB, not BGR)
3. Use MediaPipe's AI models to find faces and hands
4. Return bounding boxes (rectangles) showing WHERE the faces/hands are
5. Include confidence scores (how sure the AI is - 0.0 = not sure, 1.0 = very sure)

WHAT IS MediaPipe?
MediaPipe is an open-source framework by Google for building multimodal
machine learning pipelines. It's used for:
- Face detection & recognition
- Hand tracking
- Pose detection
- And many more AI tasks

WHAT ARE BOUNDING BOXES?
A bounding box is a rectangle that surrounds an object:
  (x, y)
    ↓
    +--------+ ← Top-left corner
    |        |
    | FACE   | ← Width
    |        |
    +--------+
         ↓
       Height

- x: left position (distance from left edge)
- y: top position (distance from top edge)  
- w: width (how wide the box is)
- h: height (how tall the box is)

================================================================================
"""

import cv2
import numpy as np
from typing import List
import logging

# LOGGING SETUP
# Logging is a way to print messages that help you debug your code
# Instead of print(), we use logger so we can control when messages show up
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from .detection_model import Detection

# STEP 1: Import MediaPipe
# This "try-except" block attempts to import MediaPipe
# If it fails, we handle it gracefully instead of crashing
logger.info("=" * 80)
logger.info("FACE & HAND DETECTOR INITIALIZATION")
logger.info("=" * 80)

try:
    import mediapipe as mp
    logger.info("✓ MediaPipe successfully imported")
    MEDIAPIPE_AVAILABLE = True
except ImportError as e:
    logger.error(f"✗ MediaPipe import FAILED: {e}")
    logger.error("  To fix this, run: pip install mediapipe")
    logger.error("  MediaPipe is Google's AI framework for detecting objects")
    MEDIAPIPE_AVAILABLE = False
    mp = None



class FaceHandDetector:
    """
    DETECTOR CLASS - DETAILED EXPLANATION FOR BEGINNERS
    
    WHAT IS A CLASS?
    A class is a blueprint/template for creating objects.
    Think of it like a cookie cutter - the class defines the shape,
    and each cookie is an instance of that class.
    
    WHAT DOES THIS CLASS DO?
    This class contains all the code needed to detect faces and hands
    in images. When you create a FaceHandDetector object, it loads
    the AI models and is ready to analyze images.
    
    ATTRIBUTES (Properties):
    - self.face_detector: The AI model for face detection
    - self.hand_detector: The AI model for hand detection
    - self.min_detection_confidence: Minimum confidence threshold
    
    METHODS (Functions inside the class):
    - __init__(): Initialize/setup the detector
    - detect_faces(): Find faces in an image
    - detect_hands(): Find hands in an image
    - detect(): Find both faces and hands
    - __del__(): Cleanup when done
    """
    
    def __init__(self, min_detection_confidence: float = 0.5):
        """
        CONSTRUCTOR - This is called when you create a new detector
        
        Example:
            detector = FaceHandDetector()  # This calls __init__
        
        Args:
            min_detection_confidence: How confident must the AI be?
                - 0.3 = Very lenient (reports even uncertain detections)
                - 0.5 = Balanced (default, good compromise)
                - 0.8 = Very strict (only reports very confident detections)
        
        WHAT HAPPENS INSIDE:
        1. Check if MediaPipe is available
        2. Load the face detection AI model
        3. Load the hand detection AI model
        4. Store settings for later use
        """
        
        logger.info("-" * 80)
        logger.info(f"Creating new FaceHandDetector")
        logger.info(f"  Confidence threshold: {min_detection_confidence}")
        logger.info("-" * 80)
        
        if not MEDIAPIPE_AVAILABLE:
            logger.error("CRITICAL: MediaPipe not available!")
            raise ImportError(
                "MediaPipe is not installed!\n"
                "Please run this command:\n"
                "    pip install mediapipe\n\n"
                "MediaPipe is Google's AI framework. It will download (~200MB)\n"
                "the pre-trained models for face and hand detection."
            )
        
        self.min_detection_confidence = min_detection_confidence
        
        # =====================================================================
        # STEP 1: Initialize Face Detection
        # =====================================================================
        logger.info("Step 1/2: Loading face detection model...")
        try:
            # Get the face detection module from MediaPipe
            self.mp_face = mp.solutions.face_detection
            
            # Create a face detector instance
            # model_selection=0: Short-range (best for 0-2 meters)
            # model_selection=1: Full-range (best for 0-5 meters)
            self.face_detector = self.mp_face.FaceDetection(
                model_selection=0,
                min_detection_confidence=min_detection_confidence
            )
            logger.info("✓ Face detection model loaded successfully")
            
        except Exception as e:
            logger.error(f"✗ Failed to load face detector: {e}")
            logger.error("  This is a critical error - face detection won't work")
            self.face_detector = None
        
        # =====================================================================
        # STEP 2: Initialize Hand Detection
        # =====================================================================
        logger.info("Step 2/2: Loading hand detection model...")
        try:
            # Get the hands module from MediaPipe
            self.mp_hands = mp.solutions.hands
            
            # Create a hand detector instance
            # static_image_mode=True: Process each image independently
            # static_image_mode=False: Use tracking between frames (for video)
            # max_num_hands=10: Can detect up to 10 hands at once
            self.hand_detector = self.mp_hands.Hands(
                static_image_mode=True,  # We're processing static images
                max_num_hands=10,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_detection_confidence
            )
            logger.info("✓ Hand detection model loaded successfully")
            
        except Exception as e:
            logger.error(f"✗ Failed to load hand detector: {e}")
            logger.error("  This is a critical error - hand detection won't work")
            self.hand_detector = None
        
        logger.info("=" * 80)
        logger.info("✓ DETECTOR FULLY INITIALIZED AND READY")
        logger.info("=" * 80)
    
    def detect_faces(self, image: np.ndarray) -> List[Detection]:
        """
        DETECT FACES IN AN IMAGE - STEP BY STEP EXPLANATION
        
        Args:
            image: An image as a numpy array in BGR format (from OpenCV)
                   BGR = Blue, Green, Red (OpenCV's color format)
        
        Returns:
            List of Detection objects, each representing one detected face
            
        WHAT HAPPENS INSIDE (THE PIPELINE):
        ┌─────────────────────────────────────────────────────────────────┐
        │ 1. Load image → 2. Convert color → 3. Run AI model →            │
        │    4. Parse results → 5. Convert coordinates → 6. Return        │
        └─────────────────────────────────────────────────────────────────┘
        """
        
        if not self.face_detector:
            logger.warning("Face detector is not available - cannot detect faces")
            return []
        
        try:
            # STEP 1: Get image dimensions
            # Why? We need to know how big the image is to convert coordinates
            h, w, _ = image.shape
            logger.debug(f"Input image size: {w}x{h} pixels")
            
            # STEP 2: Convert color space from BGR to RGB
            # WHY IS THIS IMPORTANT?
            # - OpenCV loads images as BGR by default (opposite of normal)
            # - MediaPipe expects RGB format
            # - If we don't convert, colors are wrong and detection might fail
            # 
            # ANALOGY: It's like talking to someone in their native language
            # instead of your own - you get better results!
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            logger.debug("Converted image from BGR → RGB format")
            
            # STEP 3: Run the face detection AI model
            # This is where the magic happens!
            # The AI analyzes every pixel and finds faces
            logger.debug("Running face detection AI model...")
            results = self.face_detector.process(rgb_image)
            logger.debug("Face detection AI completed")
            
            detections = []
            
            # STEP 4: Process the results
            # The AI returns all the faces it found
            if results.detections:
                logger.info(f"✓ Found {len(results.detections)} face(s)")
                
                # Process each detected face
                for idx, detection in enumerate(results.detections):
                    logger.debug(f"Processing face #{idx + 1}/{len(results.detections)}")
                    
                    # Get the bounding box of this face
                    # A bounding box is a rectangle around the face
                    bbox = detection.location_data.relative_bounding_box
                    
                    # STEP 5A: Convert coordinates from relative to pixel
                    # RELATIVE coordinates: 0.0 to 1.0 (percentage of image size)
                    # PIXEL coordinates: 0 to image_width/height (actual pixels)
                    # 
                    # EXAMPLE:
                    # If image is 640 pixels wide and relative x = 0.15
                    # Then pixel x = 0.15 * 640 = 96 pixels
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    # STEP 5B: Ensure coordinates don't go out of bounds
                    # Sometimes the AI might say the face extends beyond the image
                    # We clamp (limit) the values to stay within the image
                    x = max(0, x)  # x must be at least 0
                    y = max(0, y)  # y must be at least 0
                    width = min(width, w - x)  # width can't extend past image edge
                    height = min(height, h - y)  # height can't extend past image edge
                    
                    # STEP 5C: Get the confidence score
                    # Confidence = How sure is the AI that this is a face?
                    # 0.0 = not sure at all
                    # 1.0 = absolutely certain
                    confidence = detection.score[0]
                    
                    logger.debug(
                        f"  Face #{idx}: Position=({x}, {y}) "
                        f"Size={width}x{height} Confidence={confidence:.2f}"
                    )
                    
                    # STEP 6: Create Detection object
                    # This packages all the information about this face
                    # The Detection class is defined in models/detection.py
                    detections.append(Detection(
                        x=x,                          # Left edge
                        y=y,                          # Top edge
                        w=width,                       # Width of box
                        h=height,                      # Height of box
                        confidence=float(confidence),  # How sure (0-1)
                        label="Face"                   # What is this? A face!
                    ))
            else:
                logger.info("No faces detected in image")
            
            return detections
            
        except Exception as e:
            logger.error(f"ERROR in detect_faces: {e}")
            logger.error("Returning empty list (no detections)")
            import traceback
            traceback.print_exc()
            return []
    
    def detect_hands(self, image: np.ndarray) -> List[Detection]:
        """
        DETECT HANDS IN AN IMAGE - STEP BY STEP EXPLANATION
        
        Hand detection is more complex than face detection because:
        - Each hand has 21 landmarks (finger joints, palm, etc.)
        - We calculate the bounding box from these 21 points
        - We need to determine if it's a left or right hand
        
        Args:
            image: Image as numpy array (BGR format)
        
        Returns:
            List of Detection objects for each detected hand
        """
        
        if not self.hand_detector:
            logger.warning("Hand detector is not available - cannot detect hands")
            return []
        
        try:
            h, w, _ = image.shape
            logger.debug(f"Processing image {w}x{h} for hand detection")
            
            # Convert BGR → RGB (same as faces)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            logger.debug("Converted image BGR → RGB")
            
            # Run hand detection AI
            logger.debug("Running hand detection AI model...")
            results = self.hand_detector.process(rgb_image)
            logger.debug("Hand detection AI completed")
            
            detections = []
            
            # Check if hands were found
            if results.multi_hand_landmarks and results.multi_handedness:
                logger.info(f"✓ Found {len(results.multi_hand_landmarks)} hand(s)")
                
                # Process each detected hand
                for idx, (landmarks, handedness) in enumerate(
                    zip(results.multi_hand_landmarks, results.multi_handedness)
                ):
                    logger.debug(f"Processing hand #{idx + 1}")
                    
                    # ============================================================
                    # CALCULATE BOUNDING BOX FROM 21 HAND LANDMARKS
                    # ============================================================
                    # Hand landmarks are 21 points that represent different
                    # parts of the hand (finger tips, joints, palm, etc.)
                    # We find the min/max to create a bounding box
                    # 
                    # VISUAL:
                    # 0  = Wrist
                    # 1-4 = Thumb
                    # 5-8 = Index finger
                    # 9-12 = Middle finger
                    # 13-16 = Ring finger
                    # 17-20 = Pinky finger
                    
                    # Extract all x coordinates (convert from 0-1 to pixels)
                    x_coords = [lm.x * w for lm in landmarks.landmark]
                    # Extract all y coordinates
                    y_coords = [lm.y * h for lm in landmarks.landmark]
                    
                    # Find the bounding box boundaries
                    x_min = max(0, int(min(x_coords)))
                    y_min = max(0, int(min(y_coords)))
                    x_max = min(w, int(max(x_coords)))
                    y_max = min(h, int(max(y_coords)))
                    
                    # Calculate width and height
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    # Get confidence score
                    confidence = handedness.score
                    
                    # Determine if it's left or right hand
                    hand_label = "Unknown"
                    if hasattr(handedness.classification[0], 'label'):
                        hand_label = handedness.classification[0].label
                    
                    logger.debug(
                        f"  {hand_label} hand: Position=({x_min}, {y_min}) "
                        f"Size={width}x{height} Confidence={confidence:.2f}"
                    )
                    
                    # Create Detection object
                    detections.append(Detection(
                        x=x_min,
                        y=y_min,
                        w=width,
                        h=height,
                        confidence=float(confidence),
                        label=f"{hand_label} Hand"  # e.g., "Right Hand"
                    ))
            else:
                logger.info("No hands detected in image")
            
            return detections
            
        except Exception as e:
            logger.error(f"ERROR in detect_hands: {e}")
            logger.error("Returning empty list (no detections)")
            import traceback
            traceback.print_exc()
            return []
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        MAIN DETECTION METHOD - Find both faces and hands
        
        This is the method you'll use most often!
        It calls both detect_faces() and detect_hands() and combines results.
        
        PIPELINE:
        ┌──────────────────────────────────────────────────────────┐
        │ Input: One image                                         │
        │  ↓                                                       │
        │ Call detect_faces() → Get list of faces                 │
        │  ↓                                                       │
        │ Call detect_hands() → Get list of hands                 │
        │  ↓                                                       │
        │ Combine both lists                                      │
        │  ↓                                                       │
        │ Output: All detections (faces + hands)                  │
        └──────────────────────────────────────────────────────────┘
        
        Args:
            image: Image as numpy array (BGR format from OpenCV)
        
        Returns:
            Combined list of all detected faces and hands
        """
        
        logger.info("=" * 80)
        logger.info("STARTING DETECTION PROCESS")
        logger.info("=" * 80)
        
        # Call face detection
        face_detections = self.detect_faces(image)
        
        # Call hand detection
        hand_detections = self.detect_hands(image)
        
        # Combine results
        all_detections = face_detections + hand_detections
        
        # Summary
        logger.info(f"DETECTION COMPLETE: {len(face_detections)} faces + "
                   f"{len(hand_detections)} hands = {len(all_detections)} total")
        logger.info("=" * 80)
        
        return all_detections
    
    def __del__(self):
        """
        DESTRUCTOR - Cleanup when detector is deleted
        
        This method is called automatically when the detector object
        is no longer needed (e.g., program ends, object goes out of scope).
        
        WHY IS CLEANUP IMPORTANT?
        - AI models use a lot of memory
        - We should return that memory to the system
        - Otherwise, we might have "memory leaks" over time
        """
        try:
            if hasattr(self, 'face_detector') and self.face_detector:
                self.face_detector.close()
                logger.debug("✓ Face detector resources closed")
            if hasattr(self, 'hand_detector') and self.hand_detector:
                self.hand_detector.close()
                logger.debug("✓ Hand detector resources closed")
            logger.info("Detector cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
