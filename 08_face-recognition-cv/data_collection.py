import os
import cv2
from tqdm import tqdm

def capture_faces(output_dir='dataset', num_samples=5):
    """
    Capture face samples using webcam
    :param output_dir: Directory to save face samples
    :param num_samples: Number of face samples to capture per person
    """
    name = input("Enter person's name: ").lower().replace(" ", "_")
    path = os.path.join(output_dir, name)
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    print(f"Capturing {num_samples} samples for {name}. Look at the camera...")
    count = 0
    
    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            cv2.imwrite(f"{path}/{name}_{count}.jpg", face_img)
            count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"Captured: {count}/{num_samples}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Capturing Faces', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Completed capturing {num_samples} samples for {name}")

if __name__ == "__main__":
    capture_faces()