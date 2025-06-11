import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from ultralytics import YOLO
import torch

class AnimalDetectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Animal Detection System")
        self.setGeometry(100, 100, 1200, 800)

        # Load YOLO model
        self.model = YOLO('yolov8n.pt')
        
        # Define carnivorous animals
        self.carnivores = ['lion', 'tiger', 'bear', 'wolf', 'leopard', 'cheetah']
        
        self.setup_ui()

    def setup_ui(self):
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # Left panel for controls
        left_panel = QVBoxLayout()
        
        # Buttons
        self.image_btn = QPushButton("Load Image")
        self.video_btn = QPushButton("Load Video")
        self.detect_btn = QPushButton("Detect Animals")
        
        self.image_btn.clicked.connect(self.load_image)
        self.video_btn.clicked.connect(self.load_video)
        self.detect_btn.clicked.connect(self.detect_animals)
        
        left_panel.addWidget(self.image_btn)
        left_panel.addWidget(self.video_btn)
        left_panel.addWidget(self.detect_btn)
        
        # Results text area
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        left_panel.addWidget(QLabel("Detection Results:"))
        left_panel.addWidget(self.results_text)
        
        # Add left panel to main layout
        left_container = QWidget()
        left_container.setLayout(left_panel)
        left_container.setMaximumWidth(300)
        layout.addWidget(left_container)
        
        # Right panel for image/video display
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.display_label)
        
        # Initialize variables
        self.current_image = None
        self.video_path = None
        self.is_video_playing = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video_frame)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", 
                                                 "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            self.video_path = None
            self.current_image = cv2.imread(file_path)
            self.display_image(self.current_image)

    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", 
                                                 "Video Files (*.mp4 *.avi *.mov)")
        if file_path:
            self.video_path = file_path
            self.cap = cv2.VideoCapture(self.video_path)
            self.is_video_playing = True
            self.timer.start(30)  # 30ms refresh rate

    def update_video_frame(self):
        if self.is_video_playing and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_image = frame
                self.detect_animals()
            else:
                self.cap.release()
                self.timer.stop()
                self.is_video_playing = False

    def display_image(self, image):
        if image is not None:
            height, width = image.shape[:2]
            scale = min(800/width, 600/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
            
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.display_label.setPixmap(pixmap)

    def detect_animals(self):
        if self.current_image is not None:
            # Make a copy of the image for drawing
            image_copy = self.current_image.copy()
            
            # Run detection
            results = self.model(image_copy)
            
            carnivore_count = 0
            detected_animals = []
            
            # Process results
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Get class name
                    cls = result.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    
                    if cls in ['cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 
                              'zebra', 'giraffe', 'bird']:  # Add more animals as needed
                        color = (0, 0, 255) if cls in self.carnivores else (0, 255, 0)
                        if cls in self.carnivores:
                            carnivore_count += 1
                        
                        # Draw bounding box
                        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
                        
                        # Add label
                        label = f'{cls} {conf:.2f}'
                        cv2.putText(image_copy, label, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        detected_animals.append(f"{cls} ({conf:.2f})")
            
            # Display results
            self.display_image(image_copy)
            
            # Update results text
            results_text = f"Detected Animals:\n" + "\n".join(detected_animals)
            results_text += f"\n\nCarnivorous Animals: {carnivore_count}"
            self.results_text.setText(results_text)
            
            # Show carnivore popup if needed
            if carnivore_count > 0:
                QMessageBox.warning(self, "Warning", 
                                  f"Detected {carnivore_count} carnivorous animals!")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AnimalDetectionGUI()
    window.show()
    sys.exit(app.exec_())