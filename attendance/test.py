import threading
import time
import dlib
import mediapipe as mp
import cv2
import face_recognition
import numpy as np
import os
import sys
import django
from PIL import Image
import torch
from torch import nn
import torchvision.transforms as transforms
from torchvision import models


# Get the project root directory (one level up from this script)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add project root to Python path
sys.path.append(BASE_DIR)

# Set Django settings module
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "attendance_system.settings")  # Change to your actual project name

# Initialize Django
django.setup()

from attendance.models import Camera


# Set constants
KNOWN_FACES_DIR = 'media/known_faces'  # Path to store known faces
TOLERANCE = 0.5  # Threshold for face matching

# Global variables
known_face_encodings = []
known_face_names = []
employee_numbers = []  # List to store employee IDs

DEBUG = True  # Set to False in production

# Load the model
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, 'model', 'training_resnet50.pth')

model = models.resnet50(weights='IMAGENET1K_V1')

# Adjust the final fully connected layer (fc) for 2 classes
num_features = model.fc.in_features  # Access the fc layer instead of classifier
model.fc = nn.Linear(num_features, 2)  # Adjust for the number of classes (2 in this case)

# Load your trained model
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True))
except FileNotFoundError:
    raise RuntimeError(f"Model file not found at {MODEL_PATH}")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

model.eval()  # Set the model to evaluation mode

# Verify CUDA usage in PyTorch
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    print(f"CUDA is available! Running on GPU: {device_name} ({num_gpus} GPU(s) detected)")
else:
    print("CUDA is NOT available. Running on CPU.")


# Verify DLIB CUDA usage
if dlib.cuda.get_num_devices() > 0 and dlib.DLIB_USE_CUDA:
    print(f"DLIB is using CUDA for acceleration. ({dlib.cuda.get_num_devices()} GPU(s) detected)")
else:
    print("DLIB is NOT using CUDA. Running on CPU.")

# Move model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define your image transformations
data_transforms = transforms.Compose([
    transforms.CenterCrop(224),  # Center crop to 224x224 (or another size)
    transforms.Resize((224, 224)),  # Resize if needed
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

def load_known_faces():
    """Load known faces from the directory."""
    global known_face_encodings, known_face_names, employee_numbers

    # Check if the directory exists, if not, create it
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
        print(f"Directory '{KNOWN_FACES_DIR}' created for storing known faces.")

    # Loop through directories and load face encodings
    for name in os.listdir(KNOWN_FACES_DIR):
        person_folder = os.path.join(KNOWN_FACES_DIR, name)
        if os.path.isdir(person_folder):
            try:
                # Extract employee ID and name safely
                employee_id, full_name = name.split(" - ", 1)
                print(f"Processing Employee ID: {employee_id}, Full Name: {full_name}")

                for filename in os.listdir(person_folder):
                    image_path = os.path.join(person_folder, filename)
                    try:
                        # Load the image and get encoding
                        image = face_recognition.load_image_file(image_path)
                        encodings = face_recognition.face_encodings(image)

                        if encodings:  # Only append if encodings are found
                            known_face_encodings.append(encodings[0])
                            known_face_names.append(full_name)
                            employee_numbers.append(employee_id)
                        else:
                            print(f"No encodings found in image: {image_path}")
                    except Exception as e:
                        print(f"Error loading image: {image_path} - {e}")
            except ValueError:
                print(f"Skipping folder '{name}': it does not match expected format.")

def detect_face_spoof(frame):
    # Convert OpenCV frame to PIL Image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Apply transformations
    image_tensor = data_transforms(image).unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)  # Move tensor to the appropriate device

    # Perform inference
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)

    # Map prediction to class label
    class_idx = predicted.item()
    confidence = torch.nn.functional.softmax(outputs, dim=1)[0][class_idx].item()

    message = 'Real' if class_idx == 1 else 'Fake'

    return class_idx, confidence, message

# Mediapipe Initialization
mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

def face_detection(frame, result):
    """Processes a video frame with Mediapipe and recognizes faces using face_recognition."""
    employee_number = None  # Initialize employee_number to avoid UnboundLocalError
    try:
        # Convert to RGB (Mediapipe expects RGB input)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform face detection with Mediapipe
        results = mp_face_detection.process(rgb_frame)

        # If faces are detected
        if results.detections:
            
            for detection in results.detections:
                # Extract bounding box
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                bbox = [
                    int(bboxC.xmin * w),
                    int(bboxC.ymin * h),
                    int(bboxC.width * w),
                    int(bboxC.height * h)
                ]
                
                # Ensure the bounding box is within the frame boundaries
                face_roi = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]

                if face_roi.size == 0:  # Skip invalid bounding boxes
                    continue

                # Recognize the face encoding from the cropped region
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                
                if result == "Real":
                    #Use Face Recognition to recognize the face and get employee number
                    employee_number = recognize_face(frame, face_rgb, bbox)
                
                if result == "Fake":
                    # Draw bounding box
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)

                    # Draw background for the text with contrast
                    cv2.rectangle(frame, (bbox[0], bbox[1] - 25), (bbox[0] + bbox[2], bbox[1]), (255, 0, 0), cv2.FILLED)

                    # Overlay the name in white text
                    cv2.putText(frame, "Spoof", (bbox[0] + 6, bbox[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame, employee_number
    
    except Exception as e:
        print(f"Error during Mediapipe + recognition processing: {e}")
        return frame, None

#Summary - Face Recognition
def recognize_face(frame, face_frame, bbox):
    name = ""
    employee_number = None  # Initialize employee_number to avoid UnboundLocalError
    encodings = face_recognition.face_encodings(face_frame, model="large")

    if encodings:
        # Compare detected face encoding to known encodings
        match_results = face_recognition.compare_faces(known_face_encodings, encodings[0], TOLERANCE)
        
        if True in match_results:
            first_match_index = match_results.index(True)
            name = known_face_names[first_match_index]
            employee_number = employee_numbers[first_match_index]
            print(f"Recognized: {name} (Employee ID: {employee_number})")
        else:
            name = "Unknown"

        # Draw bounding box
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)

        # Draw background for the text with contrast
        cv2.rectangle(frame, (bbox[0], bbox[1] - 25), (bbox[0] + bbox[2], bbox[1]), (255, 0, 0), cv2.FILLED)

        # Overlay the name in white text
        cv2.putText(frame, name, (bbox[0] + 6, bbox[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return employee_number  # Will be None if no match was found

def process_camera(camera):
    """Process the camera feed."""
    if DEBUG:
        cap = cv2.VideoCapture(0) # Webcam
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            print(f"Error: Unable to access the webcam.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Failed to capture frame from webcam")
                break

            # Spoof detection
            _, _, is_real_message = detect_face_spoof(frame)

            # Face recognition
            processed_frame, employee_number = face_detection(frame, is_real_message)

            # ✅ You can now log attendance here
            if employee_number:
                print(f"🔔 Log attendance for employee ID: {employee_number}")

            # Show with unique window name
            cv2.imshow("CCTV FEED", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyWindow(f"{camera.name} - {camera.mode}")

    if not DEBUG:
        current_mode = camera.mode  # Track the initial mode
        print(f"Opening camera: {camera.name} | Mode: {camera.mode}")
        
        # Function to initialize the camera stream
        def initialize_camera():
            cap = cv2.VideoCapture(camera.camera_url)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                print(f"Error: Unable to access the camera {camera.name}.")
                return None

            return cap

        # Initialize the camera
        cap = initialize_camera()
        if not cap:
            return
        
        window_created = False
        window_name = f"{camera.name} - {current_mode}"

        while True:
            camera.refresh_from_db()

            if not camera.is_active:
                print(f"[INFO] Camera '{camera.name}' is deactivated. Waiting for reactivation...")
                cap.release()
                if window_created:
                    cv2.destroyWindow(window_name)
                    window_created = False

                while not camera.is_active:
                    camera.refresh_from_db()
                    time.sleep(1)

                # Reactivated
                print(f"[INFO] Camera '{camera.name}' reactivated. Restarting...")
                cap = initialize_camera()
                if not cap:
                    continue  # Skip and wait for next reactivation

                current_mode = camera.mode
                window_name = f"{camera.name} - {current_mode}"
                window_created = False  # Reset for new window

            ret, frame = cap.read()
            if not ret:
                print(f"Error: Failed to capture frame from {camera.name}.")
                break
            
            #Check if the camera mode has changed
            # If the mode has changed, update the window name and recreate the window
            if camera.mode != current_mode:
                print(f"[INFO] {camera.name} Mode has changed: {current_mode} → {camera.mode}")
                if window_created:
                    cv2.destroyWindow(window_name)
                current_mode = camera.mode
                window_name = f"{camera.name} - {current_mode}"
                window_created = False

            # Spoof detection
            _, _, is_real_message = detect_face_spoof(frame)

            # Face recognition
            processed_frame, employee_number = face_detection(frame, is_real_message)

            # ✅ You can now log attendance here
            if employee_number:
                print(f"🔔 Log attendance for employee ID: {employee_number}")
                # TODO: Call your attendance logging function here based on camera.mode

            # Show window (only create once)
            if not window_created:
                cv2.namedWindow(window_name)
                window_created = True
            cv2.imshow(window_name, processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if window_created:
            cv2.destroyWindow(window_name)


def main():
    print("Loading known faces...")
    load_known_faces()

    cameras = Camera.objects.filter(is_active=True)
    threads = []

    # Start a thread for each camera
    for camera in cameras:
        t = threading.Thread(target=process_camera, args=(camera,))
        t.start()
        threads.append(t)

    # Wait for all threads to finish
    for t in threads:
        t.join()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()