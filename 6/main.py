import cv2
import numpy as np
import os
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model


# Load YOLOv8 model
def load_models():
    # YOLO model for object detection (vehicles and pedestrians)
    object_detector = YOLO('yolov8n.pt')  
    
    # For color classification, we'll use a custom model
    return object_detector



def create_color_classifier():
    # Base model (MobileNetV2 is lightweight and works well for this task)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Add classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(5, activation='softmax')(x)  # 5 color classes: blue, red, black, white, other
    
    # Create model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze base layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Compile
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# In a real implementation, you would train this model on car color data
# For simplicity, we'll assume it's pre-trained and just make predictions
color_names = ['blue', 'red', 'black', 'white', 'other']

def predict_color(car_image):
    # Resize to model input size
    img = cv2.resize(car_image, (224, 224))
    img = img / 255.0  # Normalize
    
    # For demonstration, we'll use a simplified approach
    # In a real system, we'd use the trained model
    hsv = cv2.cvtColor(car_image, cv2.COLOR_BGR2HSV)
    
    # Extract dominant color
    h, s, v = cv2.split(hsv)
    h_mean = np.mean(h)
    s_mean = np.mean(s)
    
    # Simple rule-based classification
    if 100 <= h_mean <= 140 and s_mean > 50:  # Blue range in HSV
        return 'blue', 0  # 0 is the index for blue
    elif (h_mean <= 10 or h_mean >= 170) and s_mean > 50:  # Red range in HSV
        return 'red', 1
    elif s_mean < 30 and v < 100:  # Black
        return 'black', 2
    elif s_mean < 30 and v > 150:  # White
        return 'white', 3
    else:
        return 'other', 4


def process_image(image_path, object_detector):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return None, "Error loading image"
    
    # Make a copy for display
    display_img = img.copy()
    
    # Run YOLOv8 detection
    results = object_detector(img)
    
    # Initialize counters
    car_count = 0
    person_count = 0
    
    # Process each detected object
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get class, coordinates and confidence
            cls_id = int(box.cls.item())
            conf = box.conf.item()
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            class_name = object_detector.names[cls_id]
            
            # For vehicles (car, truck, bus, motorcycle)
            if class_name in ['car', 'truck', 'bus', 'motorcycle'] and conf > 0.5:
                car_count += 1
                
                # Extract car image for color classification
                car_img = img[y1:y2, x1:x2]
                if car_img.size > 0:  # Ensure we have a valid crop
                    color_name, color_id = predict_color(car_img)
                    
                    # Set rectangle color based on car color
                    if color_name == 'blue':
                        rect_color = (255, 0, 0)  # Red rectangle for blue cars (BGR format)
                    else:
                        rect_color = (0, 0, 255)  # Blue rectangle for other cars
                    
                    # Draw rectangle
                    cv2.rectangle(display_img, (x1, y1), (x2, y2), rect_color, 2)
                    
                    # Add label
                    label = f"{class_name} ({color_name})"
                    cv2.putText(display_img, label, (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, rect_color, 2)
            
            # For pedestrians
            elif class_name == 'person' and conf > 0.5:
                person_count += 1
                
                # Draw rectangle for people
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for people
                
                # Add label
                cv2.putText(display_img, "person", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Add summary text
    summary_text = f"Cars: {car_count} | People: {person_count}"
    cv2.putText(display_img, summary_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    return display_img, summary_text


class TrafficAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Analyzer")
        self.root.geometry("900x700")
        
        # Load models
        self.object_detector = load_models()
        
        # Create GUI elements
        self.create_widgets()
        
        # Current image path
        self.current_image_path = None
        
    def create_widgets(self):
        # Top frame for buttons
        top_frame = Frame(self.root)
        top_frame.pack(pady=10)
        
        # Load Image button
        load_btn = Button(top_frame, text="Load Image", command=self.load_image, 
                         font=("Arial", 12), padx=10, pady=5)
        load_btn.pack(side=tk.LEFT, padx=10)
        
        # Process Image button
        process_btn = Button(top_frame, text="Process Image", command=self.process_current_image,
                            font=("Arial", 12), padx=10, pady=5)
        process_btn.pack(side=tk.LEFT, padx=10)
        
        # Middle frame for image display
        self.image_frame = Frame(self.root, bg="gray", width=800, height=500)
        self.image_frame.pack(pady=10)
        self.image_frame.pack_propagate(False)
        
        # Image label
        self.image_label = Label(self.image_frame, bg="gray")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Bottom frame for results
        bottom_frame = Frame(self.root)
        bottom_frame.pack(pady=10, fill=tk.X, padx=20)
        
        # Results label
        self.results_label = Label(bottom_frame, text="No image processed yet",
                                 font=("Arial", 14), wraplength=850)
        self.results_label.pack()
        
        # Legend frame
        legend_frame = Frame(self.root)
        legend_frame.pack(pady=5, fill=tk.X, padx=20)
        
        # Legend labels
        Label(legend_frame, text="Legend:", font=("Arial", 12, "bold")).pack(anchor="w")
        Label(legend_frame, text="• Red rectangle: Blue car", font=("Arial", 10)).pack(anchor="w")
        Label(legend_frame, text="• Blue rectangle: Other color car", font=("Arial", 10)).pack(anchor="w")
        Label(legend_frame, text="• Green rectangle: Person", font=("Arial", 10)).pack(anchor="w")
    
    def load_image(self):
        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            self.current_image_path = file_path
            
            # Load and display image
            image = Image.open(file_path)
            image = self.resize_image(image, 800, 500)
            photo = ImageTk.PhotoImage(image)
            
            self.image_label.configure(image=photo)
            self.image_label.image = photo
            
            self.results_label.configure(text=f"Loaded: {os.path.basename(file_path)}")
    
    def process_current_image(self):
        if self.current_image_path:
            # Process the image
            processed_img, summary = process_image(self.current_image_path, self.object_detector)
            
            if processed_img is not None:
                # Convert OpenCV image to PIL format
                processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(processed_img)
                
                # Resize and display
                pil_img = self.resize_image(pil_img, 800, 500)
                photo = ImageTk.PhotoImage(pil_img)
                
                self.image_label.configure(image=photo)
                self.image_label.image = photo
                
                self.results_label.configure(text=summary)
            else:
                self.results_label.configure(text="Error processing image: " + summary)
        else:
            self.results_label.configure(text="Please load an image first.")
    
    def resize_image(self, image, width, height):
        # Calculate aspect ratio
        aspect_ratio = image.width / image.height
        
        if image.width > width or image.height > height:
            if image.width / width > image.height / height:
                # Width is the limiting factor
                new_width = width
                new_height = int(width / aspect_ratio)
            else:
                # Height is the limiting factor
                new_height = height
                new_width = int(height * aspect_ratio)
                
            return image.resize((new_width, new_height), Image.LANCZOS)
        
        return image

def main():
    root = tk.Tk()
    app = TrafficAnalyzerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()