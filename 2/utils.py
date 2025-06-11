import cv2
import numpy as np

def preprocess_image(image, target_size=(640, 640)):
    """Preprocess image for YOLO model"""
    return cv2.resize(image, target_size)

def draw_predictions(image, predictions, class_names, confidence_threshold=0.5):
    """Draw bounding boxes and labels on image"""
    image_copy = image.copy()
    
    if predictions is not None:
        for pred in predictions:
            if pred[4] >= confidence_threshold:
                x1, y1, x2, y2 = map(int, pred[:4])
                class_id = int(pred[5])
                confidence = pred[4]
                
                # Draw box
                cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label
                label = f'{class_names[class_id]} {confidence:.2f}'
                cv2.putText(image_copy, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image_copy

def get_video_properties(video_path):
    """Get video properties"""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    return {
        'width': width,
        'height': height,
        'fps': fps,
        'frame_count': frame_count
    }