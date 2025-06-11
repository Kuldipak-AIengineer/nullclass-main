import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                           QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QProgressBar)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from qt_material import apply_stylesheet

# Import DeepFace for face analysis
from deepface import DeepFace

class AnalysisWorker(QThread):
    """Worker thread to handle analysis without freezing the UI"""
    finished = pyqtSignal(dict)
    
    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path
        
    def run(self):
        try:
            results = self.analyze_image(self.image_path)
            self.finished.emit(results)
        except Exception as e:
            self.finished.emit({'error': f'Error during analysis: {str(e)}'})
    
    def analyze_image(self, image_path):
        """Analyze the image and return results based on nationality"""
        try:
            # Load image with cv2 for DeepFace
            img = cv2.imread(image_path)
            if img is None:
                return {'error': 'Failed to load image'}
                
            # Use DeepFace for analyzing face attributes
            results = {}
            
            # Check if face exists in the image
            try:
                face_objs = DeepFace.extract_faces(img_path=image_path, 
                                                 enforce_detection=False,
                                                 align=True)
                
                if not face_objs or len(face_objs) == 0:
                    return {'error': 'No face detected in the image'}
            except Exception as e:
                return {'error': f'Face detection failed: {str(e)}'}
            
            # Analyze race/ethnicity (for nationality)
            try:
                race_analysis = DeepFace.analyze(img, 
                                               actions=['race'], 
                                               enforce_detection=False)
                
                if isinstance(race_analysis, list):
                    race_analysis = race_analysis[0]
                    
                ethnicity_scores = race_analysis['race']
                
                # Map ethnicity to nationality (simplified mapping)
                ethnicity_nationality_map = {
                    'indian': 'Indian',
                    'asian': 'Asian',
                    'white': 'United States',  # Simplification
                    'middle eastern': 'Middle Eastern',
                    'latino hispanic': 'Latin American',
                    'black': 'African'  # Simplification
                }
                
                # Get the dominant ethnicity
                dominant_ethnicity = max(ethnicity_scores, key=ethnicity_scores.get)
                
                # Map to nationality
                if dominant_ethnicity.lower() in ethnicity_nationality_map:
                    results['nationality'] = ethnicity_nationality_map[dominant_ethnicity.lower()]
                else:
                    results['nationality'] = 'Other'
            except Exception as e:
                return {'error': f'Ethnicity analysis failed: {str(e)}'}
                
            # Predict emotion (always analyzed)
            try:
                emotion_analysis = DeepFace.analyze(img, 
                                                  actions=['emotion'], 
                                                  enforce_detection=False)
                
                if isinstance(emotion_analysis, list):
                    emotion_analysis = emotion_analysis[0]
                    
                results['emotion'] = emotion_analysis['dominant_emotion'].capitalize()
            except Exception as e:
                results['emotion'] = 'Unknown'
            
            # Conditional analysis based on nationality
            if results.get('nationality') in ['Indian', 'United States']:
                # Predict age for Indians and US nationality
                try:
                    age_analysis = DeepFace.analyze(img, 
                                                  actions=['age'], 
                                                  enforce_detection=False)
                    
                    if isinstance(age_analysis, list):
                        age_analysis = age_analysis[0]
                        
                    results['age'] = age_analysis['age']
                except Exception as e:
                    results['age'] = 'Unknown'
            
            if results.get('nationality') in ['Indian', 'African']:
                # For dress color, let's use a simplified approach by analyzing the bottom half of the image
                try:
                    # Get image dimensions
                    h, w, _ = img.shape
                    
                    # Use the bottom half of the image (likely containing clothing)
                    dress_region = img[h//2:, :]
                    
                    if dress_region.size > 0:
                        # Resize for faster processing
                        dress_region_resized = cv2.resize(dress_region, (100, 100))
                        
                        # Convert to RGB for better color analysis
                        dress_rgb = cv2.cvtColor(dress_region_resized, cv2.COLOR_BGR2RGB)
                        
                        # Reshape and find unique colors
                        pixels = dress_rgb.reshape(-1, 3)
                        
                        # Simple clustering to find dominant colors
                        from sklearn.cluster import KMeans
                        num_clusters = 5
                        kmeans = KMeans(n_clusters=num_clusters, n_init=10)
                        kmeans.fit(pixels)
                        
                        # Get the dominant color (largest cluster)
                        counts = np.bincount(kmeans.labels_)
                        dominant_cluster = np.argmax(counts)
                        dominant_color_rgb = kmeans.cluster_centers_[dominant_cluster].astype(int)
                        
                        # Map RGB to color name (simplified)
                        color_names = {
                            'Red': ([150, 0, 0], [255, 100, 100]),
                            'Blue': ([0, 0, 150], [100, 100, 255]),
                            'Green': ([0, 150, 0], [100, 255, 100]),
                            'Yellow': ([150, 150, 0], [255, 255, 100]),
                            'Black': ([0, 0, 0], [50, 50, 50]),
                            'White': ([200, 200, 200], [255, 255, 255]),
                            'Purple': ([100, 0, 100], [200, 100, 200]),
                            'Orange': ([200, 100, 0], [255, 200, 100]),
                            'Brown': ([100, 50, 0], [150, 100, 50])
                        }
                        
                        # Find the closest color
                        color_name = 'Other'
                        min_distance = float('inf')
                        
                        for name, (lower, upper) in color_names.items():
                            lower = np.array(lower)
                            upper = np.array(upper)
                            
                            # Check if color is in range
                            if np.all(dominant_color_rgb >= lower) and np.all(dominant_color_rgb <= upper):
                                # Calculate distance to center of range
                                center = (np.array(lower) + np.array(upper))/2
                                distance = np.sum(np.abs(dominant_color_rgb - center))
                                
                                if distance < min_distance:
                                    min_distance = distance
                                    color_name = name
                                    
                        results['dress_color'] = color_name
                    else:
                        results['dress_color'] = 'Not detected'
                except Exception as e:
                    results['dress_color'] = f'Analysis error: {str(e)}'
            
            return results
            
        except Exception as e:
            return {'error': f'Error during analysis: {str(e)}'}


class NationalityDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle('Nationality Detection System')
        self.setGeometry(100, 100, 900, 600)
        
        # Create main layout
        main_layout = QHBoxLayout()
        
        # Left panel for input and controls
        left_panel = QVBoxLayout()
        
        # Image upload button
        self.upload_btn = QPushButton('Upload Image')
        self.upload_btn.clicked.connect(self.upload_image)
        left_panel.addWidget(self.upload_btn)
        
        # Image preview
        self.preview_label = QLabel('Image Preview')
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(400, 400)
        self.preview_label.setStyleSheet("border: 2px dashed gray;")
        left_panel.addWidget(self.preview_label)
        
        # Analyze button
        self.analyze_btn = QPushButton('Analyze Image')
        self.analyze_btn.clicked.connect(self.analyze_image)
        self.analyze_btn.setEnabled(False)
        left_panel.addWidget(self.analyze_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_panel.addWidget(self.progress_bar)
        
        # Right panel for results
        right_panel = QVBoxLayout()
        
        # Results title
        results_title = QLabel('Analysis Results')
        results_title.setAlignment(Qt.AlignCenter)
        results_title.setStyleSheet("font-size: 16px; font-weight: bold;")
        right_panel.addWidget(results_title)
        
        # Results display
        self.results_layout = QVBoxLayout()
        
        # Nationality result
        self.nationality_label = QLabel('Nationality: Not detected')
        self.nationality_label.setStyleSheet("font-size: 14px;")
        self.results_layout.addWidget(self.nationality_label)
        
        # Emotion result
        self.emotion_label = QLabel('Emotion: Not detected')
        self.emotion_label.setStyleSheet("font-size: 14px;")
        self.results_layout.addWidget(self.emotion_label)
        
        # Age result (optional)
        self.age_label = QLabel('Age: Not applicable')
        self.age_label.setStyleSheet("font-size: 14px;")
        self.results_layout.addWidget(self.age_label)
        
        # Dress color result (optional)
        self.dress_color_label = QLabel('Dress Color: Not applicable')
        self.dress_color_label.setStyleSheet("font-size: 14px;")
        self.results_layout.addWidget(self.dress_color_label)
        
        # Status message
        self.status_label = QLabel('')
        self.status_label.setStyleSheet("font-size: 12px; color: gray;")
        self.results_layout.addWidget(self.status_label)
        
        right_panel.addLayout(self.results_layout)
        right_panel.addStretch()
        
        # Add panels to main layout
        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 1)
        
        # Set the main layout
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        
        # Initialize image path
        self.image_path = None
    
    def upload_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, 'Open Image', '', 'Image Files (*.png *.jpg *.jpeg *.bmp)'
        )
        
        if file_path:
            self.image_path = file_path
            self.display_image(file_path)
            self.analyze_btn.setEnabled(True)
            
            # Reset results
            self.nationality_label.setText('Nationality: Not detected')
            self.emotion_label.setText('Emotion: Not detected')
            self.age_label.setText('Age: Not applicable')
            self.dress_color_label.setText('Dress Color: Not applicable')
            self.status_label.setText('')
    
    def display_image(self, image_path):
        # Load and display the image
        pixmap = QPixmap(image_path)
        
        # Scale pixmap to fit the preview label while maintaining aspect ratio
        pixmap = pixmap.scaled(
            self.preview_label.width(), self.preview_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        
        self.preview_label.setPixmap(pixmap)
    
    def analyze_image(self):
        if not self.image_path:
            return
        
        # Show loading status
        self.status_label.setText("Analyzing image... please wait.")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.analyze_btn.setEnabled(False)
        QApplication.processEvents()
        
        # Create and start worker thread
        self.worker = AnalysisWorker(self.image_path)
        self.worker.finished.connect(self.process_results)
        self.worker.start()
    
    def process_results(self, results):
        # Hide progress bar
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        
        if 'error' in results:
            self.status_label.setText(results['error'])
            return
            
        # Update UI with results
        self.nationality_label.setText(f"Nationality: {results.get('nationality', 'Unknown')}")
        self.emotion_label.setText(f"Emotion: {results.get('emotion', 'Unknown')}")
        
        # Update conditional results based on nationality
        nationality = results.get('nationality', '')
        
        if 'age' in results:
            self.age_label.setText(f"Age: {results['age']}")
            self.age_label.setStyleSheet("font-size: 14px; color: black;")
        else:
            self.age_label.setText("Age: Not applicable")
            self.age_label.setStyleSheet("font-size: 14px; color: gray;")
            
        if 'dress_color' in results:
            self.dress_color_label.setText(f"Dress Color: {results['dress_color']}")
            self.dress_color_label.setStyleSheet("font-size: 14px; color: black;")
        else:
            self.dress_color_label.setText("Dress Color: Not applicable")
            self.dress_color_label.setStyleSheet("font-size: 14px; color: gray;")
        
        self.status_label.setText("Analysis complete!")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme='light_blue.xml')
    window = NationalityDetectionApp()
    window.show()
    sys.exit(app.exec_())