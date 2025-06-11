import sys
import os
import cv2
import numpy as np
import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                           QRadioButton, QButtonGroup, QMessageBox, QFrame)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from qt_material import apply_stylesheet
from ultralytics import YOLO
from deepface import DeepFace


class DrowsinessDetector:
    def __init__(self):
        # Load YOLO model for person detection
        self.person_detector = YOLO('yolov8n.pt')
        
        # Load face and eye detectors from OpenCV
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Eye detection threshold
        self.EYE_THRESHOLD = 0.3
        
    def detect_people(self, image):
        results = self.person_detector(image, classes=[0])  # Class 0 is person
        return results[0].boxes.data.cpu().numpy()
    
    def detect_drowsiness(self, face_img):
        # Convert to grayscale
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Detect face more precisely
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        is_drowsy = False
        age = None
        
        if len(faces) > 0:
            # Process the first face
            (x, y, w, h) = faces[0]
            
            # Extract the region of interest for the face
            roi_gray = gray[y:y + h, x:x + w]
            
            # Detect eyes in the face region
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
            # If eyes are not detected or fewer than expected, the person might be drowsy
            if len(eyes) < 2:
                is_drowsy = True
            else:
                # Calculate eye openness using histogram analysis
                eye_openness_scores = []
                for (ex, ey, ew, eh) in eyes:
                    eye_roi = roi_gray[ey:ey + eh, ex:ex + ew]
                    if eye_roi.size > 0:
                        # Calculate histogram
                        hist = cv2.calcHist([eye_roi], [0], None, [256], [0, 256])
                        # Normalize histogram
                        hist = cv2.normalize(hist, hist).flatten()
                        # Calculate variance as a measure of eye openness
                        variance = np.var(hist)
                        eye_openness_scores.append(variance)
                
                # If average eye openness is below threshold, mark as drowsy
                if eye_openness_scores and np.mean(eye_openness_scores) < self.EYE_THRESHOLD:
                    is_drowsy = True
        
        # Calculate age using DeepFace
        try:
            age_analysis = DeepFace.analyze(face_img, actions=['age'], enforce_detection=False)
            age = age_analysis[0]['age']
        except:
            pass
        
        return is_drowsy, age


class DrowsinessDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drowsiness Detection System")
        self.setGeometry(100, 100, 1000, 600)
        
        # Initialize the detector
        self.detector = DrowsinessDetector()
        
        # Current source (image or video)
        self.current_source = None
        self.video_capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Current detection results
        self.current_frame = None
        self.processed_frame = None
        self.sleeping_count = 0
        self.sleeping_ages = []
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Left panel - controls
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        
        # Source selection
        source_label = QLabel("Select Source Type:")
        left_layout.addWidget(source_label)
        
        self.source_image_radio = QRadioButton("Image")
        self.source_video_radio = QRadioButton("Video")
        self.source_camera_radio = QRadioButton("Camera")
        
        source_group = QButtonGroup()
        source_group.addButton(self.source_image_radio)
        source_group.addButton(self.source_video_radio)
        source_group.addButton(self.source_camera_radio)
        
        left_layout.addWidget(self.source_image_radio)
        left_layout.addWidget(self.source_video_radio)
        left_layout.addWidget(self.source_camera_radio)
        
        # Source selection button
        self.select_source_btn = QPushButton("Select Source")
        self.select_source_btn.clicked.connect(self.select_source)
        left_layout.addWidget(self.select_source_btn)
        
        # Process button
        self.process_btn = QPushButton("Process")
        self.process_btn.clicked.connect(self.process_source)
        left_layout.addWidget(self.process_btn)
        
        # Stop button (for video)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        left_layout.addWidget(self.stop_btn)
        
        # Stats display
        stats_frame = QFrame()
        stats_frame.setFrameShape(QFrame.StyledPanel)
        stats_layout = QVBoxLayout()
        
        self.total_people_label = QLabel("Total People: 0")
        self.sleeping_people_label = QLabel("Sleeping People: 0")
        self.age_info_label = QLabel("Ages: -")
        
        stats_layout.addWidget(self.total_people_label)
        stats_layout.addWidget(self.sleeping_people_label)
        stats_layout.addWidget(self.age_info_label)
        
        stats_frame.setLayout(stats_layout)
        left_layout.addWidget(stats_frame)
        
        # Add stretch to push controls to the top
        left_layout.addStretch()
        
        left_panel.setLayout(left_layout)
        left_panel.setFixedWidth(200)
        
        # Right panel - display
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setText("No image/video selected")
        self.display_label.setStyleSheet("background-color: #f0f0f0;")
        
        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(self.display_label, 1)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
    def select_source(self):
        if self.source_image_radio.isChecked():
            file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)")
            if file_path:
                self.current_source = file_path
                self.current_frame = cv2.imread(file_path)
                self.display_image(self.current_frame)
                
        elif self.source_video_radio.isChecked():
            file_path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)")
            if file_path:
                self.current_source = file_path
                self.video_capture = cv2.VideoCapture(file_path)
                ret, frame = self.video_capture.read()
                if ret:
                    self.display_image(frame)
                
        elif self.source_camera_radio.isChecked():
            self.current_source = 0  # Default camera
            self.video_capture = cv2.VideoCapture(0)
            ret, frame = self.video_capture.read()
            if ret:
                self.display_image(frame)
    
    def process_source(self):
        if self.current_source is None:
            QMessageBox.warning(self, "Warning", "Please select a source first!")
            return
            
        if self.source_image_radio.isChecked():
            self.process_image()
        else:
            # For video or camera
            self.stop_btn.setEnabled(True)
            self.timer.start(30)  # Update every 30ms
    
    def process_image(self):
        if self.current_frame is None:
            return
            
        # Process the frame
        processed_frame, stats = self.process_frame(self.current_frame)
        self.processed_frame = processed_frame
        
        # Update display
        self.display_image(processed_frame)
        self.update_stats(stats)
        
        # Show popup if there are sleeping people
        if stats["sleeping_count"] > 0:
            ages_str = ", ".join([str(age) for age in stats["sleeping_ages"]])
            QMessageBox.information(
                self, 
                "Drowsiness Detected", 
                f"Detected {stats['sleeping_count']} sleeping people.\nEstimated ages: {ages_str}"
            )
    
    def update_frame(self):
        if self.video_capture is None:
            return
            
        ret, frame = self.video_capture.read()
        if not ret:
            self.stop_processing()
            return
            
        # Process the frame
        processed_frame, stats = self.process_frame(frame)
        self.processed_frame = processed_frame
        
        # Update display
        self.display_image(processed_frame)
        self.update_stats(stats)
        
        # Show popup for sleeping people in video at intervals
        # Note: We don't want to spam popups for every frame
        # This would need a more sophisticated approach in a real app
        
    def process_frame(self, frame):
        # Make a copy for drawing
        display_frame = frame.copy()
        
        # Detect people
        people_boxes = self.detector.detect_people(frame)
        
        total_people = len(people_boxes)
        sleeping_count = 0
        sleeping_ages = []
        
        # Process each detected person
        for box in people_boxes:
            x1, y1, x2, y2, conf, cls = box
            
            # Skip low confidence detections
            if conf < 0.5:
                continue
                
            # Convert to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get person region
            person_img = frame[y1:y2, x1:x2]
            
            # Skip if person region is too small
            if person_img.size == 0 or person_img.shape[0] < 50 or person_img.shape[1] < 50:
                continue
            
            # Convert to grayscale for face detection
            gray_person = cv2.cvtColor(person_img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces in the person image
            faces = self.detector.face_cascade.detectMultiScale(
                gray_person,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            face_img = None
            face_coords = None
            
            if len(faces) > 0:
                # Use the detected face
                (fx, fy, fw, fh) = faces[0]
                face_coords = (x1 + fx, y1 + fy, x1 + fx + fw, y1 + fy + fh)
                face_img = person_img[fy:fy+fh, fx:fx+fw]
            else:
                # Fallback to head region estimation if no face detected
                face_h = int((y2 - y1) * 0.3)  # Assuming face is in the top 30% of person
                face_coords = (x1, y1, x2, y1 + face_h)
                face_img = person_img[0:face_h, 0:x2-x1]
            
            # Skip if face region is too small
            if face_img is None or face_img.size == 0 or face_img.shape[0] < 30 or face_img.shape[1] < 30:
                continue
            
            # Detect drowsiness
            is_drowsy, age = self.detector.detect_drowsiness(face_img)
            
            # Draw bounding box
            color = (0, 0, 255) if is_drowsy else (0, 255, 0)  # Red if drowsy, green otherwise
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw face rectangle if detected
            if face_coords:
                cv2.rectangle(display_frame, (face_coords[0], face_coords[1]), 
                             (face_coords[2], face_coords[3]), (255, 255, 0), 1)
            
            # Add age info if drowsy
            if is_drowsy and age is not None:
                sleeping_count += 1
                sleeping_ages.append(int(age))
                label = f"Sleeping, Age: {int(age)}"
                cv2.putText(display_frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            elif is_drowsy:
                sleeping_count += 1
                label = "Sleeping"
                cv2.putText(display_frame, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        stats = {
            "total_people": total_people,
            "sleeping_count": sleeping_count,
            "sleeping_ages": sleeping_ages
        }
        
        return display_frame, stats
    
    def update_stats(self, stats):
        self.total_people_label.setText(f"Total People: {stats['total_people']}")
        self.sleeping_people_label.setText(f"Sleeping People: {stats['sleeping_count']}")
        
        if stats['sleeping_ages']:
            ages_str = ", ".join([str(age) for age in stats['sleeping_ages']])
            self.age_info_label.setText(f"Ages: {ages_str}")
        else:
            self.age_info_label.setText("Ages: -")
    
    def display_image(self, image):
        h, w, c = image.shape
        bytes_per_line = 3 * w
        q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        
        # Scale image to fit the display while maintaining aspect ratio
        pixmap = QPixmap.fromImage(q_image)
        self.display_label.setPixmap(pixmap.scaled(self.display_label.width(), self.display_label.height(), 
                                                  Qt.KeepAspectRatio))
    
    def stop_processing(self):
        self.timer.stop()
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        self.stop_btn.setEnabled(False)
    
    def closeEvent(self, event):
        self.stop_processing()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme='dark_teal.xml')
    window = DrowsinessDetectionApp()
    window.show()
    sys.exit(app.exec_())


















# import sys
# import os
# import cv2
# import numpy as np
# import torch
# from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
#                            QHBoxLayout, QLabel, QPushButton, QFileDialog, 
#                            QRadioButton, QButtonGroup, QMessageBox, QFrame)
# from PyQt5.QtGui import QImage, QPixmap
# from PyQt5.QtCore import Qt, QTimer
# from qt_material import apply_stylesheet
# from ultralytics import YOLO
# from deepface import DeepFace


# import dlib
# from scipy.spatial import distance as dist



# # class DrowsinessDetector:
# #     def __init__(self):
# #         # Load YOLO model for person detection
# #         self.person_detector = YOLO('yolov8n.pt')
        
# #         # Load face detector and landmark predictor
# #         self.face_detector = dlib.get_frontal_face_detector()
# #         # You'll need to download this file: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# #         # Extract it and provide the path below
# #         model_path = "4\models\shape_predictor_68_face_landmarks.dat"
# #         if not os.path.exists(model_path):
# #             print(f"Please download the facial landmark predictor from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
# #             print(f"Extract it and place it at: {model_path}")
# #         self.landmark_predictor = dlib.shape_predictor(model_path)
        
# #         # Eye aspect ratio threshold
# #         self.EAR_THRESHOLD = 0.2
        
# #     def detect_people(self, image):
# #         results = self.person_detector(image, classes=[0])  # Class 0 is person
# #         return results[0].boxes.data.cpu().numpy()
    
# #     def eye_aspect_ratio(self, eye):
# #         # Compute the euclidean distances between the vertical eye landmarks
# #         A = dist.euclidean(eye[1], eye[5])
# #         B = dist.euclidean(eye[2], eye[4])
        
# #         # Compute the euclidean distance between the horizontal eye landmarks
# #         C = dist.euclidean(eye[0], eye[3])
        
# #         # Compute the eye aspect ratio
# #         ear = (A + B) / (2.0 * C)
        
# #         return ear
    
# #     def detect_drowsiness(self, face_img):
# #         # Convert to grayscale
# #         gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
# #         # Detect faces
# #         faces = self.face_detector(gray)
        
# #         # If no faces detected, try full image
# #         if len(faces) == 0:
# #             # For DeepFace age estimation
# #             try:
# #                 age_analysis = DeepFace.analyze(face_img, actions=['age'], enforce_detection=False)
# #                 age = age_analysis[0]['age']
# #             except:
# #                 age = None
# #             return False, age
        
# #         is_drowsy = False
# #         age = None
        
# #         # Process the first face
# #         face = faces[0]
        
# #         # Get facial landmarks
# #         landmarks = self.landmark_predictor(gray, face)
# #         points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
        
# #         # Extract eye coordinates
# #         left_eye = points[36:42]
# #         right_eye = points[42:48]
        
# #         # Calculate eye aspect ratio
# #         left_ear = self.eye_aspect_ratio(left_eye)
# #         right_ear = self.eye_aspect_ratio(right_eye)
        
# #         # Average the eye aspect ratio
# #         ear = (left_ear + right_ear) / 2.0
        
# #         # Check if below threshold
# #         if ear < self.EAR_THRESHOLD:
# #             is_drowsy = True
        
# #         # Calculate age using DeepFace
# #         try:
# #             age_analysis = DeepFace.analyze(face_img, actions=['age'], enforce_detection=False)
# #             age = age_analysis[0]['age']
# #         except:
# #             pass
        
# #         return is_drowsy, age
# class DrowsinessDetector:
#     def __init__(self):
#         # Load YOLO model for person detection
#         self.person_detector = YOLO('yolov8n.pt')
        
#         # Load face detector
#         self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#         self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
#         # Eye detection threshold
#         self.EYE_THRESHOLD = 0.3
        
#     def detect_people(self, image):
#         results = self.person_detector(image, classes=[0])  # Class 0 is person
#         return results[0].boxes.data.cpu().numpy()
    
#     def detect_drowsiness(self, face_img):
#         # Convert to grayscale
#         gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
#         # Detect face more precisely
#         faces = self.face_cascade.detectMultiScale(
#             gray,
#             scaleFactor=1.1,
#             minNeighbors=5,
#             minSize=(30, 30)
#         )
        
#         is_drowsy = False
#         age = None
        
#         if len(faces) > 0:
#             # Process the first face
#             (x, y, w, h) = faces[0]
            
#             # Extract the region of interest for the face
#             roi_gray = gray[y:y + h, x:x + w]
            
#             # Detect eyes in the face region
#             eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
#             # If eyes are not detected or fewer than expected, the person might be drowsy
#             if len(eyes) < 2:
#                 is_drowsy = True
#             else:
#                 # Calculate eye openness using histogram analysis
#                 eye_openness_scores = []
#                 for (ex, ey, ew, eh) in eyes:
#                     eye_roi = roi_gray[ey:ey + eh, ex:ex + ew]
#                     if eye_roi.size > 0:
#                         # Calculate histogram
#                         hist = cv2.calcHist([eye_roi], [0], None, [256], [0, 256])
#                         # Normalize histogram
#                         hist = cv2.normalize(hist, hist).flatten()
#                         # Calculate variance as a measure of eye openness
#                         variance = np.var(hist)
#                         eye_openness_scores.append(variance)
                
#                 # If average eye openness is below threshold, mark as drowsy
#                 if eye_openness_scores and np.mean(eye_openness_scores) < self.EYE_THRESHOLD:
#                     is_drowsy = True
        
#         # Calculate age using DeepFace
#         try:
#             age_analysis = DeepFace.analyze(face_img, actions=['age'], enforce_detection=False)
#             age = age_analysis[0]['age']
#         except:
#             pass
        
#         return is_drowsy, age


# class DrowsinessDetectionApp(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Drowsiness Detection System")
#         self.setGeometry(100, 100, 1000, 600)
        
#         # Initialize the detector
#         self.detector = DrowsinessDetector()
        
#         # Current source (image or video)
#         self.current_source = None
#         self.video_capture = None
#         self.timer = QTimer()
#         self.timer.timeout.connect(self.update_frame)
        
#         # Current detection results
#         self.current_frame = None
#         self.processed_frame = None
#         self.sleeping_count = 0
#         self.sleeping_ages = []
        
#         # Setup UI
#         self.setup_ui()
        
#     def setup_ui(self):
#         # Main widget and layout
#         main_widget = QWidget()
#         main_layout = QHBoxLayout()
        
#         # Left panel - controls
#         left_panel = QWidget()
#         left_layout = QVBoxLayout()
        
#         # Source selection
#         source_label = QLabel("Select Source Type:")
#         left_layout.addWidget(source_label)
        
#         self.source_image_radio = QRadioButton("Image")
#         self.source_video_radio = QRadioButton("Video")
#         self.source_camera_radio = QRadioButton("Camera")
        
#         source_group = QButtonGroup()
#         source_group.addButton(self.source_image_radio)
#         source_group.addButton(self.source_video_radio)
#         source_group.addButton(self.source_camera_radio)
        
#         left_layout.addWidget(self.source_image_radio)
#         left_layout.addWidget(self.source_video_radio)
#         left_layout.addWidget(self.source_camera_radio)
        
#         # Source selection button
#         self.select_source_btn = QPushButton("Select Source")
#         self.select_source_btn.clicked.connect(self.select_source)
#         left_layout.addWidget(self.select_source_btn)
        
#         # Process button
#         self.process_btn = QPushButton("Process")
#         self.process_btn.clicked.connect(self.process_source)
#         left_layout.addWidget(self.process_btn)
        
#         # Stop button (for video)
#         self.stop_btn = QPushButton("Stop")
#         self.stop_btn.clicked.connect(self.stop_processing)
#         self.stop_btn.setEnabled(False)
#         left_layout.addWidget(self.stop_btn)
        
#         # Stats display
#         stats_frame = QFrame()
#         stats_frame.setFrameShape(QFrame.StyledPanel)
#         stats_layout = QVBoxLayout()
        
#         self.total_people_label = QLabel("Total People: 0")
#         self.sleeping_people_label = QLabel("Sleeping People: 0")
#         self.age_info_label = QLabel("Ages: -")
        
#         stats_layout.addWidget(self.total_people_label)
#         stats_layout.addWidget(self.sleeping_people_label)
#         stats_layout.addWidget(self.age_info_label)
        
#         stats_frame.setLayout(stats_layout)
#         left_layout.addWidget(stats_frame)
        
#         # Add stretch to push controls to the top
#         left_layout.addStretch()
        
#         left_panel.setLayout(left_layout)
#         left_panel.setFixedWidth(200)
        
#         # Right panel - display
#         self.display_label = QLabel()
#         self.display_label.setAlignment(Qt.AlignCenter)
#         self.display_label.setText("No image/video selected")
#         self.display_label.setStyleSheet("background-color: #f0f0f0;")
        
#         # Add panels to main layout
#         main_layout.addWidget(left_panel)
#         main_layout.addWidget(self.display_label, 1)
        
#         main_widget.setLayout(main_layout)
#         self.setCentralWidget(main_widget)
        
#     def select_source(self):
#         if self.source_image_radio.isChecked():
#             file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)")
#             if file_path:
#                 self.current_source = file_path
#                 self.current_frame = cv2.imread(file_path)
#                 self.display_image(self.current_frame)
                
#         elif self.source_video_radio.isChecked():
#             file_path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)")
#             if file_path:
#                 self.current_source = file_path
#                 self.video_capture = cv2.VideoCapture(file_path)
#                 ret, frame = self.video_capture.read()
#                 if ret:
#                     self.display_image(frame)
                
#         elif self.source_camera_radio.isChecked():
#             self.current_source = 0  # Default camera
#             self.video_capture = cv2.VideoCapture(0)
#             ret, frame = self.video_capture.read()
#             if ret:
#                 self.display_image(frame)
    
#     def process_source(self):
#         if self.current_source is None:
#             QMessageBox.warning(self, "Warning", "Please select a source first!")
#             return
            
#         if self.source_image_radio.isChecked():
#             self.process_image()
#         else:
#             # For video or camera
#             self.stop_btn.setEnabled(True)
#             self.timer.start(30)  # Update every 30ms
    
#     def process_image(self):
#         if self.current_frame is None:
#             return
            
#         # Process the frame
#         processed_frame, stats = self.process_frame(self.current_frame)
#         self.processed_frame = processed_frame
        
#         # Update display
#         self.display_image(processed_frame)
#         self.update_stats(stats)
        
#         # Show popup if there are sleeping people
#         if stats["sleeping_count"] > 0:
#             ages_str = ", ".join([str(age) for age in stats["sleeping_ages"]])
#             QMessageBox.information(
#                 self, 
#                 "Drowsiness Detected", 
#                 f"Detected {stats['sleeping_count']} sleeping people.\nEstimated ages: {ages_str}"
#             )
    
#     def update_frame(self):
#         if self.video_capture is None:
#             return
            
#         ret, frame = self.video_capture.read()
#         if not ret:
#             self.stop_processing()
#             return
            
#         # Process the frame
#         processed_frame, stats = self.process_frame(frame)
#         self.processed_frame = processed_frame
        
#         # Update display
#         self.display_image(processed_frame)
#         self.update_stats(stats)
    
#     def process_frame(self, frame):
#         # Make a copy for drawing
#         display_frame = frame.copy()
        
#         # Detect people
#         people_boxes = self.detector.detect_people(frame)
        
#         total_people = len(people_boxes)
#         sleeping_count = 0
#         sleeping_ages = []
        
#         # Process each detected person
#         for box in people_boxes:
#             x1, y1, x2, y2, conf, cls = box
            
#             # Skip low confidence detections
#             if conf < 0.5:
#                 continue
                
#             # Get person region
#             person_img = frame[int(y1):int(y2), int(x1):int(x2)]
            
#             # Skip if person region is too small
#             if person_img.size == 0 or person_img.shape[0] < 50 or person_img.shape[1] < 50:
#                 continue
            
#             # Face detection using face detector (better approach)
#             gray = cv2.cvtColor(person_img, cv2.COLOR_BGR2GRAY)
#             faces = self.detector.face_detector(gray)
            
#             if len(faces) > 0:
#                 # Use the detected face
#                 face = faces[0]
#                 face_x1 = int(x1) + face.left()
#                 face_y1 = int(y1) + face.top()
#                 face_x2 = int(x1) + face.right()
#                 face_y2 = int(y1) + face.bottom()
                
#                 face_img = frame[face_y1:face_y2, face_x1:face_x2]
#             else:
#                 # Fallback to head region estimation if no face detected
#                 face_y1 = int(y1)
#                 face_y2 = int(y1 + (y2-y1) * 0.3)  # Assuming face is in the top 30% of person
#                 face_x1 = int(x1)
#                 face_x2 = int(x2)
                
#                 face_img = frame[face_y1:face_y2, face_x1:face_x2]
            
#             # Skip if face region is too small
#             if face_img.size == 0 or face_img.shape[0] < 30 or face_img.shape[1] < 30:
#                 continue
            
#             # Detect drowsiness
#             is_drowsy, age = self.detector.detect_drowsiness(face_img)
            
#             # Draw bounding box
#             color = (0, 0, 255) if is_drowsy else (0, 255, 0)  # Red if drowsy, green otherwise
#             cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
#             # Add age info if drowsy
#             if is_drowsy and age is not None:
#                 sleeping_count += 1
#                 sleeping_ages.append(int(age))
#                 label = f"Sleeping, Age: {int(age)}"
#                 cv2.putText(display_frame, label, (int(x1), int(y1)-10), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
#         stats = {
#             "total_people": total_people,
#             "sleeping_count": sleeping_count,
#             "sleeping_ages": sleeping_ages
#         }
        
#         return display_frame, stats
    
#     def update_stats(self, stats):
#         self.total_people_label.setText(f"Total People: {stats['total_people']}")
#         self.sleeping_people_label.setText(f"Sleeping People: {stats['sleeping_count']}")
        
#         if stats['sleeping_ages']:
#             ages_str = ", ".join([str(age) for age in stats['sleeping_ages']])
#             self.age_info_label.setText(f"Ages: {ages_str}")
#         else:
#             self.age_info_label.setText("Ages: -")
    
#     def display_image(self, image):
#         h, w, c = image.shape
#         bytes_per_line = 3 * w
#         q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        
#         # Scale image to fit the display while maintaining aspect ratio
#         pixmap = QPixmap.fromImage(q_image)
#         self.display_label.setPixmap(pixmap.scaled(self.display_label.width(), self.display_label.height(), 
#                                                   Qt.KeepAspectRatio))
    
#     def stop_processing(self):
#         self.timer.stop()
#         if self.video_capture:
#             self.video_capture.release()
#             self.video_capture = None
#         self.stop_btn.setEnabled(False)
    
#     def closeEvent(self, event):
#         self.stop_processing()
#         event.accept()


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     apply_stylesheet(app, theme='dark_teal.xml')
#     window = DrowsinessDetectionApp()
#     window.show()
#     sys.exit(app.exec_())