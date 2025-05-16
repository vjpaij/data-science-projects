import cv2
import face_recognition
import pickle
import numpy as np
from datetime import datetime

class FaceRecognizer:
    def __init__(self, encodings_file='encodings.pkl', tolerance=0.6, detection_model='hog', process_every=1):
        """
        Initialize face recognizer
        :param encodings_file: Path to saved encodings
        :param tolerance: Distance tolerance for face matching (lower is stricter)

        Optimization:
        Frame resizing: Process smaller frames for detection
        Face detection model selection: Use 'hog' for CPU, 'cnn' for GPU
        Skip frames: Process every nth frame if needed
        Multi-threading: Separate face detection and recognition
        """
        self.detection_model = detection_model
        self.process_every = process_every
        self.frame_count = 0
        self.tolerance = tolerance
        self.load_encodings(encodings_file)
        self.known_faces = list(set(self.data["names"]))
        self.attendance = set()
        
    def load_encodings(self, encodings_file):
        """Load pre-computed face encodings"""
        with open(encodings_file, 'rb') as f:
            self.data = pickle.load(f)
        print(f"Loaded encodings for {len(self.data['names'])} faces")
    
    def recognize_faces(self, frame):
        """Recognize faces in a frame"""
        # Optimized face recognition
        self.frame_count += 1
        if self.frame_count % self.process_every != 0:
            return [], []
        
        # Convert to RGB (face_recognition uses RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect face locations and encodings
        boxes = face_recognition.face_locations(rgb, model='hog')
        encodings = face_recognition.face_encodings(rgb, boxes)
        
        names = []
        
        for encoding in encodings:
            # Compare with known encodings
            matches = face_recognition.compare_faces(self.data["encodings"], 
                                                   encoding, 
                                                   tolerance=self.tolerance)
            name = "Unknown"
            
            if True in matches:
                matched_idxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                
                for i in matched_idxs:
                    name = self.data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                
                name = max(counts, key=counts.get)
            
            names.append(name)
            
            # Record attendance
            if name != "Unknown" and name not in self.attendance:
                self.attendance.add(name)
                self.record_attendance(name)
        
        return boxes, names
    
    def record_attendance(self, name):
        """Record attendance with timestamp"""
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
        with open('attendance.csv', 'a') as f:
            f.write(f"{name},{dt_string}\n")
    
    def run(self):
        """Run real-time face recognition"""
        video_capture = cv2.VideoCapture(0)
        
        while True:
            ret, frame = video_capture.read()
            
            if not ret:
                continue
                
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            
            # Recognize faces
            boxes, names = self.recognize_faces(small_frame)
            
            # Display results
            for (top, right, bottom, left), name in zip(boxes, names):
                # Scale back up face locations
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # Draw box and label
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            
            # Display attendance count
            cv2.putText(frame, f"Recognized: {len(self.attendance)}/{len(self.known_faces)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Face Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    recognizer = FaceRecognizer()
    recognizer.run()