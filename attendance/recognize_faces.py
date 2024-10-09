import torch
from torchvision import models
import torch.nn as nn
from torchvision import transforms
import face_recognition
import numpy as np
import base64
from PIL import Image
import io
import os
import cv2
from django.conf import settings  # Import settings to access MEDIA_ROOT

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = os.path.join(settings.MEDIA_ROOT, 'known_faces')  # Path to the known faces directory
MODEL_PATH = os.path.join(CURRENT_DIR, 'training.pth')
TOLERANCE = 0.5
FAKE_TOLERANCE_HIGH = 0.75  # Threshold for definitely fake
FAKE_TOLERANCE_LOW = 0.50   # Threshold for definitely real

# Define the transformation for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),                      # Randomly flip the image horizontally
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly change brightness, contrast, saturation, and hue
    transforms.RandomRotation(degrees=15),                  # Randomly rotate the image within ±15 degrees
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for MobileNet
])

# Load the Haar Cascade for eye detection
EYE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_eye.xml'
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)

# Load the Haar Cascade for face detection
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# Load known faces once and store them in a global variable
known_face_encodings = []
known_face_names = []
employee_ids = []  # List to store employee IDs

def load_known_faces(KNOWN_FACES_DIR):
    global known_face_encodings, known_face_names, employee_ids
    
    # Check if the directory exists, and if not, create it
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
        print(f"Directory '{KNOWN_FACES_DIR}' created for storing known faces.")
    
    # Proceed with loading faces if the directory exists
    for name in os.listdir(KNOWN_FACES_DIR):
        person_folder = os.path.join(KNOWN_FACES_DIR, name)
        if os.path.isdir(person_folder):
            # Split the folder name to extract employee ID and full name
            try:
                employee_id, full_name = name.split(" - ", 1)  # Split into two parts
                print(f"Processing Employee ID: {employee_id}, Full Name: {full_name}")

                for filename in os.listdir(person_folder):
                    image_path = os.path.join(person_folder, filename)
                    try:
                        image = face_recognition.load_image_file(image_path)
                        encoding = face_recognition.face_encodings(image)[0]
                        known_face_encodings.append(encoding)
                        known_face_names.append(full_name)  # Use the full name
                        employee_ids.append(employee_id)  # Store employee ID
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")
            except ValueError:
                print(f"Skipping folder '{name}': it does not match expected format.")

# Load the deepfake detection model using PyTorch
class FakeFaceDetector(nn.Module):
    def __init__(self, num_classes=2):  # Accept num_classes as a parameter
        super(FakeFaceDetector, self).__init__()
        self.model = models.mobilenet_v2(weights='IMAGENET1K_V1')  # Load MobileNetV2 with pre-trained weights
        num_features = self.model.classifier[1].in_features  # Get input features from the classifier
        self.model.classifier[1] = nn.Linear(num_features, num_classes)  # Direct replacement for final layer

    def forward(self, x):
        x = self.model(x)
        return x

def load_model():
    model = FakeFaceDetector()
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"), weights_only=True), strict=False)
        model.eval()  # Set the model to evaluation mode
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    return model

def detect_face_opencv(image_array):
    """
    Detect faces in an image using OpenCV's Haar Cascade.
    Returns the largest detected face as a PIL image or None if no face is found.
    """
    # Convert the image to grayscale for the face detector
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar Cascade
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # If faces are found, use the largest one
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])  # Get the largest face
        face_image = image_array[y:y + h, x:x + w]
        return Image.fromarray(face_image)
    else:
        return None  # No face found

def crop_face(image_array):
    """
    Detects and crops the largest face in the image.
    Returns the cropped face as a PIL image or None if no face is found.
    """
    face_locations = face_recognition.face_locations(image_array)
    
    if face_locations:
        # Use the first face (largest)
        top, right, bottom, left = face_locations[0]
        face_image = image_array[top:bottom, left:right]
        return Image.fromarray(face_image)
    else:
        return detect_face_opencv(image_array)  # No face found

def detect_fake_face(image_data):
    """
    Detect if the face in the image is fake or real by cropping the face first
    and then running the deepfake detection model.
    """
    # Decode the base64 image if not already in bytes
    if isinstance(image_data, str):
        image_data = image_data.split(",")[1]
        image_data = base64.b64decode(image_data)

    # Open the image and convert it to RGB
    image = Image.open(io.BytesIO(image_data))
    image = image.convert('RGB')
    image_array = np.array(image)

    # Crop the face from the image
    face_image = crop_face(image_array)
    if face_image is None:
        return "No face detected."

    # Transform the cropped face for the model
    face_image = transform(face_image).unsqueeze(0)

    # Load the model
    model = load_model()

    # Run the model to detect if the face is fake
    with torch.no_grad():
        output = model(face_image)
        probabilities = torch.softmax(output, dim=1)[0]  # Apply softmax to get probabilities
        fake_confidence = probabilities[1].item()  # Confidence for "fake" class
        fake_confidence = round(fake_confidence, 2)
        
    # Determine if the face is real or fake based on the confidence thresholds
    if fake_confidence >= FAKE_TOLERANCE_HIGH:
        print(f"Fake Confidence: {fake_confidence:.2f} - [Fake]")
        return "Fake"
    elif fake_confidence < FAKE_TOLERANCE_LOW:
        print(f"Fake Confidence: {fake_confidence:.2f} - [Real]")
        return "Real"
    else:
        print(f"Fake Confidence: {fake_confidence:.2f} - [Undetermined]")
        return "Undetermined"

def recognize_faces(image_data):
    fake_status = detect_fake_face(image_data)  # Check if the face is real or fake
    if fake_status == "Real":
        # Proceed to recognition
        recognition_results = recognize_faces_from_image(image_data)
        return recognition_results
    
    if fake_status == "Fake":
        print("Fake Face Detected: ", fake_status)
        return fake_status
    
    if fake_status == "Undetermined":
        print("Fake Face Detected: ", fake_status)
        return fake_status

def recognize_faces_from_image(image_data):
    try:
        # List to store results for each face
        results = []

        # Decode the base64 image if not already in bytes
        if isinstance(image_data, str):
            image_data = image_data.split(",")[1]
            image_data = base64.b64decode(image_data)

        # Open the image
        image = Image.open(io.BytesIO(image_data))

        # Convert image to RGB
        image = image.convert('RGB')
        image_array = np.array(image)

        # Find faces in the captured image
        unknown_face_encodings = face_recognition.face_encodings(image_array)

        for face_encoding in unknown_face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, TOLERANCE)
            name = "Unknown"
            employee_id = "N/A"  # Default if not recognized

            # Check if we have a match
            if True in matches:
                # Find the indexes of all matched faces
                matched_indexes = [i for (i, b) in enumerate(matches) if b]

                # Use the first matched face (could also average scores)
                first_match_index = matched_indexes[0]
                name = known_face_names[first_match_index]
                employee_id = employee_ids[first_match_index]  # Get corresponding employee ID

            results.append({"name": name, "employee_id": employee_id})  # Append result

        return results  # Return results

    except Exception as e:
        print(f"Error during recognition: {e}")
        return []

# Load known faces when the module is loaded
load_known_faces(KNOWN_FACES_DIR)
