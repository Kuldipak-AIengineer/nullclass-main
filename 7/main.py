import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
import time
from deepface import DeepFace
from collections import deque

# Initialize data storage
csv_path = 'age_emotion_log.csv'
if not os.path.exists(csv_path):
    pd.DataFrame(columns=['Timestamp', 'Age', 'Emotion', 'Status']).to_csv(csv_path, index=False)

def log_entry(age, emotion, status):
    """Log entry to CSV file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = pd.DataFrame([[timestamp, age, emotion, status]], 
                           columns=['Timestamp', 'Age', 'Emotion', 'Status'])
    new_data.to_csv(csv_path, mode='a', header=False, index=False)

def main():
    print("Initializing face detection...")
    # Initialize OpenCV's built-in Haar Cascade face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # For smoothing predictions
    age_buffer = deque(maxlen=5)
    emotion_buffer = {}
    
    print("Starting video capture...")
    # Start video capture
    cap = cv2.VideoCapture(0)
    
    # Set a smaller frame size for faster processing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract face ROI
            face = frame[y:y+h, x:x+w]
            
            if face.size == 0:
                continue
            
            try:
                # Analyze face with DeepFace
                results = DeepFace.analyze(face, actions=['age', 'emotion'], 
                                         enforce_detection=False, silent=True)
                
                if isinstance(results, list):
                    results = results[0]
                
                # Get age and emotion
                raw_age = results['age']
                
                # Use a more accurate age calculation - applying correction factors
                # based on common DeepFace accuracy patterns
                corrected_age = raw_age
                if raw_age < 10:  # DeepFace tends to overestimate child ages
                    corrected_age = max(1, raw_age * 0.8)
                elif raw_age > 60:  # DeepFace tends to underestimate senior ages
                    corrected_age = raw_age * 1.1
                
                # Add to buffer for smoothing
                age_buffer.append(corrected_age)
                smoothed_age = int(sum(age_buffer) / len(age_buffer))
                
                # Get emotion and smooth it too
                emotion = results['dominant_emotion']
                if emotion not in emotion_buffer:
                    emotion_buffer[emotion] = 1
                else:
                    emotion_buffer[emotion] += 1
                
                # Get most frequent emotion from last few frames
                smoothed_emotion = max(emotion_buffer, key=emotion_buffer.get)
                
                # Reset emotion buffer periodically to adapt to changes
                if sum(emotion_buffer.values()) > 10:
                    # Keep the top 2 emotions
                    top_emotions = sorted(emotion_buffer.items(), key=lambda x: x[1], reverse=True)[:2]
                    emotion_buffer.clear()
                    for e, count in top_emotions:
                        emotion_buffer[e] = count
                
                # Determine status and color based on age
                if smoothed_age < 13 or smoothed_age > 60:
                    status = "Not allowed"
                    color = (0, 0, 255)  # Red (BGR)
                    display_emotion = "N/A"  # Don't show emotion for restricted ages
                else:
                    status = "Allowed"
                    color = (0, 255, 0)  # Green (BGR)
                    display_emotion = smoothed_emotion
                
                # Draw rectangle and text
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                cv2.putText(frame, f"Age: {smoothed_age}", (x, y-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"Status: {status}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                if display_emotion != "N/A":
                    cv2.putText(frame, f"Emotion: {display_emotion}", (x, y+h+25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Log entry
                log_entry(smoothed_age, display_emotion, status)
                
            except Exception as e:
                print(f"Error analyzing face: {e}")
                continue
        
        # Display result
        cv2.imshow('Age and Emotion Detection', frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
    