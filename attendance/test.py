import sys
import threading
import time
import django
from django.utils.timezone import now, make_naive
import dlib
import mediapipe as mp
import cv2
import face_recognition
import numpy as np
import os
from PIL import Image
import torch
from torch import nn
import torchvision.transforms as transforms
from torchvision import models


# Set constants
KNOWN_FACES_DIR = 'media/known_faces'  # Path to store known faces
TOLERANCE = 0.4  # Threshold for face matching

# Global variables
known_face_encodings = []
known_face_names = []
employee_numbers = []  # List to store employee IDs
display_frame = None
frame_lock = threading.Lock()
stop_display_thread = False


# Get the project root directory (one level up from this script)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add project root to Python path
sys.path.append(BASE_DIR)

# Set Django settings module
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "attendance_system.settings")  # Change to your actual project name

# Initialize Django
django.setup()

today = now()
clock_in_time = make_naive(today) # This will make the datetime timezone-aware

# Import models AFTER setting up Django
from attendance.models import Employee, ShiftRecord

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
    print(f"✅ CUDA is available! Running on GPU: {device_name} ({num_gpus} GPU(s) detected)")
else:
    print("⚠️ CUDA is NOT available. Running on CPU.")


# Verify DLIB CUDA usage
if dlib.cuda.get_num_devices() > 0 and dlib.DLIB_USE_CUDA:
    print(f"✅ DLIB is using CUDA for acceleration. ({dlib.cuda.get_num_devices()} GPU(s) detected)")
    print(f"CUDA Device Name: {dlib.cuda.get_device_name(0)}")
    #Set the model to Convolutional Neural Network (CNN) for face detection - best for GPU
    face_location_model = "cnn"
else:
    print("⚠️ DLIB is NOT using CUDA. Running on CPU.")
    #Set the model to Histogram of Oriented Gradients (HOG) for face detection - best for CPU
    face_location_model = "hog"




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
mp_face_detection = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def face_detection(frame, result):
    """Processes a video frame with Mediapipe and recognizes faces using face_recognition."""
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
                    #Use Face Recognition
                    recognize_face(frame, face_rgb, bbox)
                
                if result == "Fake":
                    # Draw bounding box
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)

                    # Draw background for the text with contrast
                    cv2.rectangle(frame, (bbox[0], bbox[1] - 25), (bbox[0] + bbox[2], bbox[1]), (255, 0, 0), cv2.FILLED)

                    # Overlay the name in white text
                    cv2.putText(frame, "Spoof", (bbox[0] + 6, bbox[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame
    
    except Exception as e:
        print(f"Error during Mediapipe + recognition processing: {e}")
        return frame

#Summary - Face Recognition
def recognize_face(frame, face_frame, bbox):
    name = ""
    emp_no = ""

     # Use face_location to find faces in the frame
    face_locations = face_recognition.face_locations(face_frame, number_of_times_to_upsample=1, model=face_location_model)

    if not face_locations:
        pass  # No faces found, skip processing
    
    # Get the encodings for the detected faces
    # Note: face_recognition.face_encodings() can return multiple encodings if multiple faces are detected
    encodings = face_recognition.face_encodings(face_frame, known_face_locations=face_locations, model="large")

    # Check if any encodings were found
    if encodings:
        # Compare the encodings with known faces
        face_distances = face_recognition.face_distance(known_face_encodings, encodings[0])
        
        # Get the index of the closest match
        best_match_index = np.argmin(face_distances)

        # Check if the distance is below the tolerance threshold
        if face_distances[best_match_index] < TOLERANCE:
            name = known_face_names[best_match_index]
            emp_no = employee_numbers[best_match_index]

            # Clock in the employee if recognized
            if emp_no:
                # Check if the employee is already clocked in for today
                this_employee = Employee.objects.get(employee_number=emp_no)
                existing_record = ShiftRecord.objects.filter(employee=this_employee, date=clock_in_time).first()
                # If not clocked in, clock in the employee
                if existing_record:
                    print(f"Employee {emp_no} already clocked in successfully.")
                if not existing_record:
                    clock_in_employee(emp_no)
        else:
            name = "Unknown"

        # Draw bounding box
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)

        # Draw background for the text with contrast
        cv2.rectangle(frame, (bbox[0], bbox[1] - 25), (bbox[0] + bbox[2], bbox[1]), (255, 0, 0), cv2.FILLED)

        # Overlay the name in white text
        cv2.putText(frame, name, (bbox[0] + 6, bbox[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


#Clock-in the employee
# This function is called when a recognized employee is detected
# It creates a ShiftRecord for the employee with the current time as clock-in time  
def clock_in_employee(employee_number):
    try:
        # Check if the employee exists in the database
        employee = Employee.objects.get(employee_number=employee_number)
        # Check if the employee is already clocked in for today
        ShiftRecord.objects.create(employee=employee, clock_in=today)
        print(f"Employee {employee_number} clocked in successfully.")
    except Employee.DoesNotExist:
        print(f"Employee {employee_number} does not exist.")
    except Exception as e:
        print(f"Error clocking in employee {employee_number}: {e}")

def display_video():
    global display_frame, stop_display_thread

    while not stop_display_thread:
        if display_frame is not None:
            frame_lock.acquire()
            frame_to_show = display_frame.copy()
            frame_lock.release()

            cv2.imshow("Face Recognition with Names", frame_to_show)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_display_thread = True
            break

    cv2.destroyAllWindows()

class AsyncCamera:
    def __init__(self, source=0):
        self.capture = cv2.VideoCapture(source)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.ret = False
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            self.ret, self.frame = self.capture.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        self.thread.join()
        self.capture.release()

def main():
    global display_frame, stop_display_thread

    camera = AsyncCamera()

    if not camera.capture.isOpened():
        print("Error: Unable to access the camera.")
        return

    # Start display thread
    display_thread = threading.Thread(target=display_video)
    display_thread.start()

    print("Press 'q' to quit.")

    # FPS setup
    prev_time = time.time()
    fps = 0

    while not stop_display_thread:
        ret, frame = camera.read()
        if not ret or frame is None:
            continue

        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # Anti-spoofing detection
        _, _, is_real_message = detect_face_spoof(frame)

        # Process the frame for recognition
        processed_frame = face_detection(frame, is_real_message)

        # Draw FPS counter
        cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Send frame to display thread
        frame_lock.acquire()
        display_frame = processed_frame.copy()
        frame_lock.release()

    # Cleanup
    camera.stop()


if __name__ == "__main__":
    print("Loading known faces...")
    load_known_faces()
    main()
