import torch
from torchvision import models
from torchvision.models import ResNet18_Weights  # Import the weight enum
import torch.nn as nn
from torchvision import transforms
import face_recognition
import numpy as np
import base64
from PIL import Image
import io
import os
import cv2

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = os.path.join(CURRENT_DIR, 'known_faces')
MODEL_PATH = os.path.join(CURRENT_DIR, 'training.pth')
TOLERANCE = 0.55

# Define the transformation for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 600x600
    transforms.ToTensor()            # Convert images to tensor
])

# Load the Haar Cascade for eye detection
EYE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_eye.xml'
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)

# Load known faces once and store them in a global variable
known_face_encodings = []
known_face_names = []

def load_known_faces(known_faces_dir):
    global known_face_encodings, known_face_names
    for name in os.listdir(known_faces_dir):
        person_folder = os.path.join(known_faces_dir, name)
        if os.path.isdir(person_folder):
            for filename in os.listdir(person_folder):
                image_path = os.path.join(person_folder, filename)
                try:
                    image = face_recognition.load_image_file(image_path)
                    encoding = face_recognition.face_encodings(image)[0]
                    known_face_encodings.append(encoding)
                    known_face_names.append(name)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

# Load the deepfake detection model using PyTorch
class FakeFaceDetector(nn.Module):
    def __init__(self, num_classes=2):  # Accept num_classes as a parameter
        super(FakeFaceDetector, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)  # Direct replacement

    def forward(self, x):
        x = self.model(x)
        return x

def load_model():
    model = FakeFaceDetector()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"), weights_only=True), strict=False)
    model.eval()  # Set the model to evaluation mode
    return model

def detect_fake_face(image_data):
    # Load and preprocess the image for the model
    image = Image.open(io.BytesIO(base64.b64decode(image_data.split(",")[1])))
    image = transform(image).unsqueeze(0)

    # Load the model
    model = load_model()

    # Run the model to detect if the face is fake
    with torch.no_grad():
        output = model(image)
        _, prediction = torch.max(output, 1)

    return "Fake" if prediction.item() == 1 else "Real"

def recognize_faces(image_data):
    fake_status = detect_fake_face(image_data)  # Check if the face is real or fake
    if fake_status == "Real":
        # Proceed to recognition
        recognition_results = recognize_faces_from_image(image_data)
        return recognition_results
    
    if fake_status == "Fake":
        print("Fake Face Detected: ", fake_status)
        return {"message": "Face detected as Fake. Skipping recognition."}

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
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                results.append({
                    "status": "Match found",
                    "name": name,
                    "distance": face_distances[best_match_index]
                })
                print(f"Recognized: {name}")
            else:
                results.append({
                    "message": "No match found",
                    "name": None
                })
                print("No match found for the detected face.")

        # Return results for all detected faces
        return results

    except Exception as e:
        print(f"Error processing image: {e}")
        return {"error": f"Error processing image: {str(e)}"}

# Call this function when your application starts
load_known_faces(KNOWN_FACES_DIR)
